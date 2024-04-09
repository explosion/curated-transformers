from dataclasses import dataclass

import torch

from ...layers.activations import Activation
from ..config import (
    RotaryEmbeddingConfig,
    TransformerAttentionLayerConfig,
    TransformerConfig,
    TransformerEmbeddingLayerConfig,
    TransformerFeedForwardLayerConfig,
    TransformerLayerConfig,
)


@dataclass
class FalconConfig(TransformerConfig):
    """
    Falcon (`Penedo et al., 2019`_) model configuration.

    .. _Penedo et al., 2019: https://arxiv.org/abs/2306.01116
    """

    new_decoder_architecture: bool

    def __init__(
        self,
        *,
        attention_probs_dropout_prob: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        hidden_dropout_prob: float = 0.0,
        hidden_width: int = 2560,
        layer_norm_eps: float = 1e-5,
        new_decoder_architecture: bool = False,
        n_query_heads: int = 71,
        n_key_value_heads: int = 1,
        n_hidden_layers: int = 32,
        rotary_embedding_base: int = 10000,
        rotary_embedding_fraction: float = 0.25,
        use_alibi: bool = False,
        use_bias: bool = False,
        use_parallel_attention: bool = True,
        n_pieces: int = 50280,
    ):
        """
        :param attention_probs_dropout_prob:
            Dropout to apply after attention.
        :param dtype:
            Data type to use for model parameters.
        :param hidden_dropout_prob:
            Dropout to apply to the hidden and embedding layers.
        :param hidden_width:
            Hidden width of the transformer.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param n_query_heads:
            Number of query heads.
        :param n_key_value_heads:
            Number of key and value heads.
        :param n_hidden_layers:
            Number of hidden layers.
        :param rotary_embedding_base:
            Base in signifying the rotary embedding period.
        :param rotary_embedding_fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        :param use_alibi:
            Use ALiBi linear biases in self-attention.
        :param use_bias:
            Use bias in linear layers.
        :param use_parallel_attention:
            Use parallel attention.
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
                n_query_heads=n_query_heads,
                n_key_value_heads=n_key_value_heads,
                use_parallel_attention=use_parallel_attention,
                rotary_embeddings=RotaryEmbeddingConfig(
                    rotary_fraction=rotary_embedding_fraction,
                    rotary_base=rotary_embedding_base,
                ),
                use_bias=use_bias,
                use_alibi=use_alibi,
            ),
            feedforward=TransformerFeedForwardLayerConfig(
                hidden_width=hidden_width,
                intermediate_width=4 * hidden_width,
                activation=Activation.GELU,
                use_bias=use_bias,
                use_gate=False,
            ),
            dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            n_hidden_layers=n_hidden_layers,
        )
        self.dtype = dtype
        self.new_decoder_architecture = new_decoder_architecture
