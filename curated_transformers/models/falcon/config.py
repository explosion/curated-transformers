from dataclasses import dataclass


@dataclass
class FalconAttentionConfig:
    """
    `Falcon`_ attention configuration.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    dropout_prob: float
    hidden_width: int
    multi_query: bool
    num_attention_heads: int
    rotary_fraction: float
    rotary_base: int
    use_bias: bool

    def __init__(
        self,
        *,
        dropout_prob: float = 0.1,
        hidden_width: int = 2560,
        multi_query: bool = True,
        num_attention_heads: int = 71,
        rotary_base=10000,
        rotary_fraction=0.25,
        use_bias: bool = False,
    ):
        """
        :param dropout_prob:
            Dropout to apply after attention.
        :param hidden_width:
            Hidden width of the transformer.
        :param multi_query:
            Use multiple query heads and single key and value heads.
        :param num_attention_heads:
            Number of attention heads.
        :param rotary_base:
            Base in signifying the rotary embedding period.
        :param rotary_fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        :param use_bias:
            Use bias in linear layers.
        """

        self.dropout_prob = dropout_prob
        self.hidden_width = hidden_width
        self.multi_query = multi_query
        self.num_attention_heads = num_attention_heads
        self.rotary_base = rotary_base
        self.rotary_fraction = rotary_fraction
        self.use_bias = use_bias


@dataclass
class FalconEmbeddingConfig:
    """
    `Falcon`_ embedding configuration.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    dropout_prob: float
    embedding_width: int
    layer_norm_eps: float
    vocab_size: int

    def __init__(
        self,
        *,
        dropout_prob: float = 0.1,
        embedding_width: int = 2560,
        layer_norm_eps: float = 1e-5,
        vocab_size: int = 50432,
    ) -> None:
        """
        :param dropout_prob:
            Dropout to apply after attention.
        :param embedding_width:
            Width of the embeddings.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param vocab_size:
            Vocabulary size (number of embeddings).
        """

        self.dropout_prob = dropout_prob
        self.embedding_width = embedding_width
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps


@dataclass
class FalconLayerConfig:
    """
    `Falcon`_ layer configuration.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    dropout_prob: float
    hidden_width: int
    layer_norm_eps: float
    num_hidden_layers: int
    use_bias: bool

    def __init__(
        self,
        *,
        dropout_prob: float = 0.0,
        hidden_width: int = 2560,
        num_hidden_layers: int = 32,
        layer_norm_eps: float = 1e-5,
        use_bias: bool = False,
    ) -> None:
        """
        :param dropout_prob:
            Dropout to apply after hidden layers.
        :param hidden_width:
            Hidden width of the transformer.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param num_hidden_layers:
            Number of hidden layers.
        :param use_bias:
            Use bias in linear layers.
        """

        self.dropout_prob = dropout_prob
        self.hidden_width = hidden_width
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias


class FalconConfig:
    """
    `Falcon`_ model configuration.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    attention: FalconAttentionConfig
    embedding: FalconEmbeddingConfig
    layer: FalconLayerConfig

    def __init__(
        self,
        *,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        hidden_width: int = 2560,
        layer_norm_eps: float = 1e-5,
        multi_query: bool = True,
        num_attention_heads: int = 71,
        num_hidden_layers: int = 32,
        rotary_embedding_base: int = 10000,
        rotary_embedding_fraction: float = 0.25,
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
        :param multi_query:
            Use multiple query heads and single key and value heads.
        :param num_hidden_layers:
            Number of hidden layers.
        :param rotary_embedding_base:
            Base in signifying the rotary embedding period.
        :param rotary_embedding_fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        :param use_bias:
            Use bias in linear layers.
        :param vocab_size:
            Vocabulary size (number of embeddings).
        """

        # TODO: max_position_embeddings and model_max_length are currently
        #       not used. We may want to limit the rotary embeddings to these
        #       values in the future. We should check empirically if the auto
        #       resizing in rotary embeddings makes sense.

        self.attention = FalconAttentionConfig(
            dropout_prob=attention_probs_dropout_prob,
            hidden_width=hidden_width,
            multi_query=multi_query,
            num_attention_heads=num_attention_heads,
            rotary_fraction=rotary_embedding_fraction,
            rotary_base=rotary_embedding_base,
            use_bias=use_bias,
        )
        self.embedding = FalconEmbeddingConfig(
            dropout_prob=hidden_dropout_prob,
            embedding_width=hidden_width,
            vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps,
        )
        self.layer = FalconLayerConfig(
            dropout_prob=hidden_dropout_prob,
            hidden_width=hidden_width,
            layer_norm_eps=layer_norm_eps,
            num_hidden_layers=num_hidden_layers,
        )
