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
class LlamaConfig(TransformerConfig):
    """
    Llama (`Touvron et al., 2023 [a]`_, `Touvron et al., 2023 [b]`_) model configuration.

    .. _Touvron et al., 2023 [a]: https://arxiv.org/abs/2302.13971
    .. _Touvron et al., 2023 [b]: https://arxiv.org/abs/2307.09288
    """

    def __init__(
        self,
        *,
        attention_probs_dropout_prob: float = 0.0,
        activation: Activation = Activation.GELU,
        dtype: torch.dtype = torch.float16,
        hidden_dropout_prob: float = 0.0,
        hidden_width: int = 2560,
        intermediate_width: int = 10240,
        rms_norm_eps: float = 1e-5,
        n_query_heads: int = 32,
        n_hidden_layers: int = 32,
        n_key_value_heads: int = 32,
        rotary_embedding_base: int = 10000,
        rotary_embedding_fraction: float = 0.25,
        n_pieces: int = 50280,
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
        :param intermediate_width:
            Intermediate width in the feed-forward layer.
            The non-linearity is applied in this intermediate width.
        :param rms_norm_eps:
            Epsilon for layer normalization.
        :param n_query_heads:
            Number of query heads.
        :param n_hidden_layers:
            Number of hidden layers.
        :param n_key_value_heads:
            Number of key-value heads.
        :param rotary_embedding_base:
            Base in signifying the rotary embedding period.
        :param rotary_embedding_fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        :param n_pieces:
            Vocabulary size (number of embeddings).
        """

        self.embedding = TransformerEmbeddingLayerConfig(
            dropout_prob=hidden_dropout_prob,
            embedding_width=hidden_width,
            n_pieces=n_pieces,
            layer_norm_eps=rms_norm_eps,
            n_positions=None,
            n_types=None,
        )
        self.layer = TransformerLayerConfig(
            attention=TransformerAttentionLayerConfig(
                dropout_prob=attention_probs_dropout_prob,
                hidden_width=hidden_width,
                n_query_heads=n_query_heads,
                n_key_value_heads=n_key_value_heads,
                rotary_embeddings=RotaryEmbeddingConfig(
                    rotary_fraction=rotary_embedding_fraction,
                    rotary_base=rotary_embedding_base,
                ),
                use_alibi=False,
                use_bias=False,
                use_parallel_attention=False,
            ),
            feedforward=TransformerFeedForwardLayerConfig(
                hidden_width=hidden_width,
                intermediate_width=intermediate_width,
                activation=activation,
                use_bias=False,
                use_gate=True,
            ),
            dropout_prob=hidden_dropout_prob,
            layer_norm_eps=rms_norm_eps,
            n_hidden_layers=n_hidden_layers,
        )
        self.dtype = dtype
