from dataclasses import dataclass


@dataclass
class FalconAttentionConfig:
    """
    `Falcon`_ attention configuration.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    dropout_prob: float
    hidden_width: int
    num_query_heads: int
    num_key_value_heads: int
    parallel_attention: bool
    rotary_fraction: float
    rotary_base: int
    use_bias: bool

    def __init__(
        self,
        *,
        dropout_prob: float = 0.1,
        hidden_width: int = 2560,
        num_query_heads: int = 71,
        num_key_value_heads: int = 1,
        parallel_attention: bool = True,
        rotary_base=10000,
        rotary_fraction=0.25,
        use_bias: bool = False,
    ):
        """
        :param dropout_prob:
            Dropout to apply after attention.
        :param hidden_width:
            Hidden width of the transformer.
        :param num_query_heads:
            Number of attention heads.
        :param num_key_value_heads:
            Number of key and value heads.
        :param parallel_attention:
            Use parallel attention.
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
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.parallel_attention = parallel_attention
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


@dataclass
class FalconConfig:
    """
    `Falcon`_ model configuration.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    attention: FalconAttentionConfig
    embedding: FalconEmbeddingConfig
    layer: FalconLayerConfig
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
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            parallel_attention=parallel_attention,
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
        self.new_decoder_architecture = new_decoder_architecture
