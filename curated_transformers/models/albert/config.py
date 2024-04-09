from dataclasses import dataclass

import torch

from ...layers.activations import Activation
from ..config import (
    TransformerAttentionLayerConfig,
    TransformerConfig,
    TransformerEmbeddingLayerConfig,
    TransformerFeedForwardLayerConfig,
    TransformerLayerConfig,
)


@dataclass
class ALBERTLayerConfig(TransformerLayerConfig):
    """
    ALBERT (`Lan et al., 2022`_) layer configuration.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    n_layers_per_group: int
    n_hidden_groups: int

    def __init__(
        self, *args, n_layers_per_group: int = 1, n_hidden_groups: int = 1, **kwargs
    ):
        """
        :param n_layers_per_group:
            Number of layers per layer group.
        :param n_hidden_groups:
            Number of hidden groups.
        """
        super(ALBERTLayerConfig, self).__init__(*args, **kwargs)
        self.n_layers_per_group = n_layers_per_group
        self.n_hidden_groups = n_hidden_groups


@dataclass
class ALBERTConfig(TransformerConfig):
    """
    ALBERT (`Lan et al., 2022`_) model configuration.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    layer: ALBERTLayerConfig

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        embedding_width: int = 128,
        hidden_width: int = 768,
        n_layers_per_group: int = 1,
        intermediate_width: int = 3072,
        n_attention_heads: int = 12,
        n_hidden_layers: int = 12,
        n_hidden_groups: int = 1,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        activation: Activation = Activation.GELUNew,
        n_pieces: int = 30000,
        n_types: int = 2,
        n_positions: int = 512,
        model_max_length: int = 512,
        layer_norm_eps: float = 1e-12,
    ):
        """
        :param dtype:
            Data type to use for model parameters.
        :param embedding_width:
            Width of the embedding representations.
        :param hidden_width:
            Width of the transformer hidden layers.
        :param n_layers_per_group:
            Number of layers per layer group.
        :param intermediate_width:
            Width of the intermediate projection layer in the
            point-wise feed-forward layer.
        :param n_attention_heads:
            Number of self-attention heads.
        :param n_hidden_layers:
            Number of hidden layers.
        :param n_hidden_groups:
            Number of hidden groups.
        :param attention_probs_dropout_prob:
            Dropout probabilty of the self-attention layers.
        :param hidden_dropout_prob:
            Dropout probabilty of the point-wise feed-forward and
            embedding layers.
        :param activation:
            Activation used by the pointwise feed-forward layers.
        :param n_pieces:
            Size of main vocabulary.
        :param n_types:
            Size of token type vocabulary.
        :param n_positions:
            Maximum length of position embeddings.
        :param model_max_length:
            Maximum length of model inputs.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        """
        self.embedding = TransformerEmbeddingLayerConfig(
            embedding_width=embedding_width,
            n_pieces=n_pieces,
            n_types=n_types,
            n_positions=n_positions,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )
        self.layer = ALBERTLayerConfig(
            attention=TransformerAttentionLayerConfig(
                hidden_width=hidden_width,
                dropout_prob=attention_probs_dropout_prob,
                n_key_value_heads=n_attention_heads,
                n_query_heads=n_attention_heads,
                rotary_embeddings=None,
                use_alibi=False,
                use_bias=True,
                use_parallel_attention=False,
            ),
            feedforward=TransformerFeedForwardLayerConfig(
                hidden_width=hidden_width,
                intermediate_width=intermediate_width,
                activation=activation,
                use_bias=True,
                use_gate=False,
            ),
            n_hidden_layers=n_hidden_layers,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
            n_layers_per_group=n_layers_per_group,
            n_hidden_groups=n_hidden_groups,
        )
        self.dtype = dtype
        self.model_max_length = model_max_length
