from dataclasses import dataclass

# Defaults taken from syntaxdot
# https://github.com/tensordot/syntaxdot/blob/22bd3d43ed2d7fcbef8a6217b01684194fae713f/syntaxdot-transformers/src/models/bert/config.rs#L25


@dataclass
class BERTEmbeddingConfig:
    """
    BERT (`Devlin et al., 2018`_) embedding configuration.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    embedding_width: int
    vocab_size: int
    type_vocab_size: int
    max_position_embeddings: int
    layer_norm_eps: float
    dropout_prob: float

    def __init__(
        self,
        *,
        embedding_width: int = 768,
        vocab_size: int = 30000,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        dropout_prob: float = 0.1,
    ) -> None:
        """
        :param embedding_width:
            Width of the embedding representations.
        :param vocab_size:
            Size of main vocabulary.
        :param type_vocab_size:
            Size of token type vocabulary.
        :param max_position_embeddings:
            Maximum length of position embeddings.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param dropout_prob:
            Dropout probabilty for the embedding layer.
        """
        self.embedding_width = embedding_width
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout_prob = dropout_prob


@dataclass
class BERTAttentionConfig:
    """
    BERT (`Devlin et al., 2018`_) attention configuration.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    hidden_width: int
    num_attention_heads: int
    dropout_prob: float

    def __init__(
        self,
        *,
        hidden_width: int = 768,
        num_attention_heads: int = 12,
        dropout_prob: float = 0.1,
    ) -> None:
        """
        :param hidden_width:
            Width of the projections for query, key and value.
        :param num_attention_heads:
            Number of self-attention heads.
        :param dropout_prob:
            Dropout probabilty for self-attention.
        """
        self.hidden_width = hidden_width
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob


@dataclass
class BERTLayerConfig:
    """
    BERT (`Devlin et al., 2018`_) layer configuration.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    hidden_width: int
    intermediate_width: int
    num_hidden_layers: int
    hidden_act: str
    layer_norm_eps: float
    dropout_prob: float

    def __init__(
        self,
        *,
        hidden_width: int = 768,
        intermediate_width: int = 3072,
        num_hidden_layers: int = 12,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        dropout_prob: float = 1.0,
    ) -> None:
        """
        :params hidden_width:
            Hidden width of the transformer.
        :params intermediate_width:
            Intermediate width in the feed-forward layer.
        :params num_hidden_layers:
            Number of hidden layers.
        :params hidden_act:
            Activation used by the feed-forward layers.
            Applied on the intermediate representation.
        :params layer_norm_eps:
            Epsilon for layer normalization.
        :params dropout_prob:
            Dropout probabilty to apply after hidden layers.
        """
        self.hidden_width = hidden_width
        self.intermediate_width = intermediate_width
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout_prob = dropout_prob


@dataclass
class BERTConfig:
    """
    BERT (`Devlin et al., 2018`_) model configuration.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    embedding: BERTEmbeddingConfig
    attention: BERTAttentionConfig
    layer: BERTLayerConfig
    model_max_length: int
    padding_id: int

    def __init__(
        self,
        *,
        embedding_width: int = 768,
        hidden_width: int = 768,
        intermediate_width: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "gelu",
        vocab_size: int = 30000,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        model_max_length: int = 512,
        layer_norm_eps: float = 1e-12,
        padding_id: int = 0,
    ):
        """
        :param embedding_width:
            Width of the embedding representations.
        :param hidden_width:
            Width of the transformer hidden layers.
        :param intermediate_width:
            Width of the intermediate projection layer in the
            point-wise feed-forward layer.
        :param num_attention_heads:
            Number of self-attention heads.
        :param num_hidden_layers:
            Number of hidden layers.
        :param attention_probs_dropout_prob:
            Dropout probabilty of the self-attention layers.
        :param hidden_dropout_prob:
            Dropout probabilty of the point-wise feed-forward and
            embedding layers.
        :param hidden_act:
            Activation used by the point-wise feed-forward layers.
        :param vocab_size:
            Size of main vocabulary.
        :param type_vocab_size:
            Size of token type vocabulary.
        :param max_position_embeddings:
            Maximum length of position embeddings.
        :param model_max_length:
            Maximum length of model inputs.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param padding_id:
            Index of the padding meta-token.
        """
        self.embedding = BERTEmbeddingConfig(
            embedding_width=embedding_width,
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )
        self.attention = BERTAttentionConfig(
            hidden_width=hidden_width,
            num_attention_heads=num_attention_heads,
            dropout_prob=attention_probs_dropout_prob,
        )
        self.layer = BERTLayerConfig(
            hidden_width=hidden_width,
            intermediate_width=intermediate_width,
            num_hidden_layers=num_hidden_layers,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )
        self.model_max_length = model_max_length
        self.padding_id = padding_id
