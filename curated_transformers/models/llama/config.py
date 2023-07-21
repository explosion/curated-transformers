from dataclasses import dataclass

from ..config import (
    RotaryEmbeddingConfig,
    TransformerAttentionConfig,
    TransformerEmbeddingConfig,
    TransformerLayerConfig,
)


class LLaMAAttentionConfig(TransformerAttentionConfig):
    """
    LLaMA (`Touvron et al., 2023`_) attention configuration.

    .. _Touvron et al., 2023: https://arxiv.org/abs/2302.13971
    """

    ...


class LLaMAEmbeddingConfig(TransformerEmbeddingConfig):
    """
    LLaMA (`Touvron et al., 2023`_) embedding configuration.

    .. _Touvron et al., 2023: https://arxiv.org/abs/2302.13971
    """

    ...


class LLaMALayerConfig(TransformerLayerConfig):
    """
    LLaMA (`Touvron et al., 2023`_) layer configuration.

    .. _Touvron et al., 2023: https://arxiv.org/abs/2302.13971
    """

    ...


class LLaMAConfig:
    """
    LLaMA (`Touvron et al., 2023`_) model configuration.

    .. _Touvron et al., 2023: https://arxiv.org/abs/2302.13971
    """

    attention: LLaMAAttentionConfig
    embedding: LLaMAEmbeddingConfig
    layer: LLaMALayerConfig

    def __init__(
        self,
        *,
        attention_probs_dropout_prob: float = 0.0,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        hidden_width: int = 2560,
        intermediate_width: int = 10240,
        rms_norm_eps: float = 1e-5,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        rotary_embedding_base: int = 10000,
        rotary_embedding_fraction: float = 0.25,
        vocab_size: int = 50280,
    ):
        """
        :param attention_probs_dropout_prob:
            Dropout to apply after attention.
        :param hidden_act:
            Activation in the feed-forward layer. See
            :class:`curated_transformers.layers.feedforward.PointwiseFeedForward`
            for possible values.
        :param hidden_dropout_prob:
            Dropout to apply to the hidden and embedding layers.
        :param hidden_width:
            Hidden width of the transformer.
        :param intermediate_width:
            Intermediate width in the feed-forward layer.
            The non-linearity is applied in this intermediate width.
        :param rms_norm_eps:
            Epsilon for layer normalization.
        :param num_attention_heads:
            Number of attention heads.
        :param num_hidden_layers:
            Number of hidden layers.
        :param rotary_embedding_base:
            Base in signifying the rotary embedding period.
        :param rotary_embedding_fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        :param vocab_size:
            Vocabulary size (number of embeddings).
        """
        self.attention = LLaMAAttentionConfig(
            dropout_prob=attention_probs_dropout_prob,
            hidden_width=hidden_width,
            num_query_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            parallel_attention=False,
            rotary_embeddings=RotaryEmbeddingConfig(
                rotary_fraction=rotary_embedding_fraction,
                rotary_base=rotary_embedding_base,
            ),
            use_bias=False,
            use_alibi=False,
        )
        self.embedding = LLaMAEmbeddingConfig(
            dropout_prob=hidden_dropout_prob,
            embedding_width=hidden_width,
            vocab_size=vocab_size,
            layer_norm_eps=rms_norm_eps,
            max_position_embeddings=None,
            type_vocab_size=None,
        )
        self.layer = LLaMALayerConfig(
            dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            hidden_width=hidden_width,
            intermediate_width=intermediate_width,
            layer_norm_eps=rms_norm_eps,
            num_hidden_layers=num_hidden_layers,
            use_bias=False,
        )
