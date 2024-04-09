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
class BERTConfig(TransformerConfig):
    """
    BERT (`Devlin et al., 2018`_) model configuration.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    model_max_length: int

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        embedding_width: int = 768,
        hidden_width: int = 768,
        intermediate_width: int = 3072,
        n_attention_heads: int = 12,
        n_hidden_layers: int = 12,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        activation: Activation = Activation.GELU,
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
        :param intermediate_width:
            Width of the intermediate projection layer in the
            point-wise feed-forward layer.
        :param n_attention_heads:
            Number of self-attention heads.
        :param n_hidden_layers:
            Number of hidden layers.
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
        self.layer = TransformerLayerConfig(
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
        )
        self.dtype = dtype
        self.model_max_length = model_max_length
