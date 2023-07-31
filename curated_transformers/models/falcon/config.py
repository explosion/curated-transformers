from dataclasses import dataclass

from ...layers.activations import Activation
from ..config import (
    RotaryEmbeddingConfig,
    TransformerAttentionLayerConfig,
    TransformerEmbeddingLayerConfig,
    TransformerFeedForwardLayerConfig,
    TransformerLayerConfig,
)


@dataclass
class FalconConfig:
    """
    Falcon (`Penedo et al., 2019`_) model configuration.

    .. _Penedo et al., 2019: https://arxiv.org/abs/2306.01116
    """

    embedding: TransformerEmbeddingLayerConfig
    layer: TransformerLayerConfig
    new_decoder_architecture: bool

    def __init__(
        self,
        *,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        hidden_width: int = 2560,
        layer_norm_eps: float = 1e-5,
        new_decoder_architecture: bool = False,
        num_query_heads: int = 71,
        num_key_value_heads: int = 1,
        num_hidden_layers: int = 32,
        parallel_attention: bool = True,
        rotary_embedding_base: int = 10000,
        rotary_embedding_fraction: float = 0.25,
        use_alibi: bool = False,
        use_bias: bool = False,
        vocab_size: int = 50280,
    ):
        """
        :param attention_probs_dropout_prob:
            Dropout to apply after attention.
        :param hidden_dropout_prob:
            Dropout to apply to the hidden and embedding layers.
        :param hidden_width:
            Hidden width of the transformer.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param num_query_heads:
            Number of query heads.
        :param num_key_value_heads:
            Number of key and value heads.
        :param num_hidden_layers:
            Number of hidden layers.
        :param parallel_attention:
            Use parallel attention.
        :param rotary_embedding_base:
            Base in signifying the rotary embedding period.
        :param rotary_embedding_fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        :param use_alibi:
            Use ALiBi linear biases in self-attention.
        :param use_bias:
            Use bias in linear layers.
        :param vocab_size:
            Vocabulary size (number of embeddings).
        """

        # TODO: max_position_embeddings and model_max_length are currently
        #       not used. We may want to limit the rotary embeddings to these
        #       values in the future. We should check empirically if the auto
        #       resizing in rotary embeddings makes sense.

        self.embedding = TransformerEmbeddingLayerConfig(
            dropout_prob=hidden_dropout_prob,
            embedding_width=hidden_width,
            vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=None,
            type_vocab_size=None,
        )
        self.layer = TransformerLayerConfig(
            attention=TransformerAttentionLayerConfig(
                dropout_prob=attention_probs_dropout_prob,
                hidden_width=hidden_width,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                parallel_attention=parallel_attention,
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
            num_hidden_layers=num_hidden_layers,
        )
        self.new_decoder_architecture = new_decoder_architecture
