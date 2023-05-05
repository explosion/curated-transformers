from dataclasses import dataclass

from ..bert import BertConfig, BertAttentionConfig, BertEmbeddingConfig
from ..bert import BertLayerConfig


@dataclass
class AlbertLayerConfig(BertLayerConfig):
    inner_group_num: int
    num_hidden_groups: int

    def __init__(self, *args, inner_group_num=1, num_hidden_groups=1, **kwargs):
        super(AlbertLayerConfig, self).__init__(*args, **kwargs)
        self.inner_group_num = inner_group_num
        self.num_hidden_groups = num_hidden_groups


@dataclass
class AlbertConfig(BertConfig):
    layer: AlbertLayerConfig

    def __init__(
        self,
        *,
        embedding_width: int = 128,
        hidden_width: int = 768,
        inner_group_num: int = 1,
        intermediate_width: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        num_hidden_groups: int = 1,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        hidden_act: str = "gelu_new",
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
        self.layer = AlbertLayerConfig(
            hidden_width=hidden_width,
            inner_group_num=inner_group_num,
            intermediate_width=intermediate_width,
            num_hidden_layers=num_hidden_layers,
            num_hidden_groups=num_hidden_groups,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )
        self.model_max_length = model_max_length
        self.padding_idx = padding_idx
