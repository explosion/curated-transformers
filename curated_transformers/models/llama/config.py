from dataclasses import dataclass


@dataclass
class LLaMAAttentionConfig:
    """
    LLaMA (`Touvron et al., 2023`_) attention configuration.

    .. _Touvron et al., 2023: https://arxiv.org/abs/2302.13971
    """

    dropout_prob: float
    hidden_width: int
    num_attention_heads: int
    rotary_fraction: float
    rotary_base: int

    def __init__(
        self,
        *,
        dropout_prob: float = 0.0,
        hidden_width: int = 2560,
        num_attention_heads: int = 32,
        rotary_base=10000,
        rotary_fraction=0.25,
    ):
        """
        :param dropout_prob:
            Dropout to apply after attention.
        :param hidden_width:
            Hidden width of the transformer.
        :param num_attention_heads:
            Number of attention heads.
        :param rotary_base:
            Base in signifying the rotary embedding period.
        :param rotary_fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        """

        self.dropout_prob = dropout_prob
        self.hidden_width = hidden_width
        self.num_attention_heads = num_attention_heads
        self.rotary_base = rotary_base
        self.rotary_fraction = rotary_fraction


@dataclass
class LLaMAEmbeddingConfig:
    """
    LLaMA (`Touvron et al., 2023`_) embedding configuration.

    .. _Touvron et al., 2023: https://arxiv.org/abs/2302.13971
    """

    dropout_prob: float
    embedding_width: int
    rms_norm_eps: float
    vocab_size: int

    def __init__(
        self,
        *,
        dropout_prob: float = 0.0,
        embedding_width: int = 2560,
        rms_norm_eps: float = 1e-5,
        vocab_size: int = 50432,
    ) -> None:
        """
        :param dropout_prob:
            Dropout to apply after attention.
        :param embedding_width:
            Width of the embeddings.
        :param rms_norm_eps:
            Epsilon for layer normalization.
        :param vocab_size:
            Vocabulary size (number of embeddings).
        """

        self.dropout_prob = dropout_prob
        self.embedding_width = embedding_width
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps


@dataclass
class LLaMALayerConfig:
    """
    LLaMA (`Touvron et al., 2023`_) layer configuration.

    .. _Touvron et al., 2023: https://arxiv.org/abs/2302.13971
    """

    dropout_prob: float
    hidden_act: str
    hidden_width: int
    intermediate_width: int
    rms_norm_eps: float
    num_hidden_layers: int

    def __init__(
        self,
        *,
        dropout_prob: float = 0.0,
        hidden_width: int = 2560,
        intermediate_width: int = 10240,
        num_hidden_layers: int = 32,
        hidden_act: str = "gelu",
        rms_norm_eps: float = 1e-5,
    ) -> None:
        """
        :param dropout_prob:
            Dropout to apply after hidden layers.
        :param hidden_act:
            Activation in the feed-forward layer. See
            :class:`curated_transformers.layers.feedforward.PointwiseFeedForward`
            for possible values.
        :param hidden_width:
            Hidden width of the transformer.
        :param intermediate_width:
            Intermediate width in the feed-forward layer.
            The non-linearity is applied in this intermediate width.
        :param rms_norm_eps:
            Epsilon for layer normalization.
        :param num_hidden_layers:
            Number of hidden layers.
        """

        self.dropout_prob = dropout_prob
        self.hidden_width = hidden_width
        self.intermediate_width = intermediate_width
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps


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
            num_attention_heads=num_attention_heads,
            rotary_fraction=rotary_embedding_fraction,
            rotary_base=rotary_embedding_base,
        )
        self.embedding = LLaMAEmbeddingConfig(
            dropout_prob=hidden_dropout_prob,
            embedding_width=hidden_width,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
        )
        self.layer = LLaMALayerConfig(
            dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            hidden_width=hidden_width,
            intermediate_width=intermediate_width,
            rms_norm_eps=rms_norm_eps,
            num_hidden_layers=num_hidden_layers,
        )
