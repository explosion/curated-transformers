from dataclasses import dataclass


# Defaults taken from syntaxdot
# https://github.com/tensordot/syntaxdot/blob/22bd3d43ed2d7fcbef8a6217b01684194fae713f/syntaxdot-transformers/src/models/bert/config.rs#L25


@dataclass
class BertEmbeddingConfig:
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
        self.embedding_width = embedding_width
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout_prob = dropout_prob


@dataclass
class BertAttentionConfig:
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
        self.hidden_width = hidden_width
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob


@dataclass
class BertLayerConfig:
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
        self.hidden_width = hidden_width
        self.intermediate_width = intermediate_width
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout_prob = dropout_prob


@dataclass
class BertConfig:
    embedding: BertEmbeddingConfig
    attention: BertAttentionConfig
    layer: BertLayerConfig
    model_max_length: int
    padding_idx: int

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
        padding_idx: int = 0,
    ):
        self.embedding = BertEmbeddingConfig(
            embedding_width=embedding_width,
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )
        self.attention = BertAttentionConfig(
            hidden_width=hidden_width,
            num_attention_heads=num_attention_heads,
            dropout_prob=attention_probs_dropout_prob,
        )
        self.layer = BertLayerConfig(
            hidden_width=hidden_width,
            intermediate_width=intermediate_width,
            num_hidden_layers=num_hidden_layers,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )
        self.model_max_length = model_max_length
        self.padding_idx = padding_idx
