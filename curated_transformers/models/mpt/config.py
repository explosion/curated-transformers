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
class MPTConfig(TransformerConfig):
    """
    `MosaicML MPT`_ model configuration.

    .. _MosaicML MPT: https://www.mosaicml.com/blog/mpt-7b
    """

    def __init__(
        self,
        *,
        attention_probs_dropout_prob: float = 0.0,
        activation: Activation = Activation.GELU,
        dtype: torch.dtype = torch.bfloat16,
        hidden_dropout_prob: float = 0.0,
        hidden_width: int = 4096,
        intermediate_width_multiplier: int = 4,
        layer_norm_eps: float = 1e-5,
        model_max_length: int = 2048,
        n_attention_heads: int = 32,
        n_hidden_layers: int = 32,
        n_pieces: int = 50432,
        use_bias=False
    ):
        """
        :param attention_probs_dropout_prob:
            Dropout to apply after attention.
        :param activation:
            Activation used by the pointwise feed-forward layers.
        :param dtype:
            Data type to use for model parameters.
        :param hidden_dropout_prob:
            Dropout to apply to the hidden and embedding layers.
        :param hidden_width:
            Hidden width of the transformer.
        :param intermediate_width_multiplier:
            Multiplier for the intermediate width. The hidden width is
            multiplied by this value to get the intermediate width.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param model_max_length:
            Maximum sequence length of the model.
        :param n_attention_heads:
            Number of attention heads.
        :param n_hidden_layers:
            Number of hidden layers.
        :param n_pieces:
            Vocabulary size (number of embeddings).
        """

        # TODO: n_positions and model_max_length are currently
        #       not used. We may want to limit the rotary embeddings to these
        #       values in the future. We should check empirically if the auto
        #       resizing in rotary embeddings makes sense.

        self.embedding = TransformerEmbeddingLayerConfig(
            dropout_prob=hidden_dropout_prob,
            embedding_width=hidden_width,
            n_pieces=n_pieces,
            layer_norm_eps=layer_norm_eps,
            n_positions=None,
            n_types=None,
        )
        self.layer = TransformerLayerConfig(
            attention=TransformerAttentionLayerConfig(
                dropout_prob=attention_probs_dropout_prob,
                hidden_width=hidden_width,
                n_query_heads=n_attention_heads,
                n_key_value_heads=n_attention_heads,
                rotary_embeddings=None,
                use_alibi=True,
                use_bias=use_bias,
                use_parallel_attention=False,
            ),
            feedforward=TransformerFeedForwardLayerConfig(
                hidden_width=hidden_width,
                intermediate_width=intermediate_width_multiplier * hidden_width,
                activation=activation,
                use_bias=use_bias,
                use_gate=False,
            ),
            dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            n_hidden_layers=n_hidden_layers,
        )
        self.dtype = dtype
        self.model_max_length = model_max_length
