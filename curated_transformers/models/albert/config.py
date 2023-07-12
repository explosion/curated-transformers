from dataclasses import dataclass

from ..bert import BERTAttentionConfig, BERTConfig, BERTEmbeddingConfig, BERTLayerConfig


@dataclass
class ALBERTLayerConfig(BERTLayerConfig):
    """
    ALBERT (`Lan et al., 2022`_) layer configuration.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    inner_group_num: int
    num_hidden_groups: int

    def __init__(
        self, *args, inner_group_num: int = 1, num_hidden_groups: int = 1, **kwargs
    ):
        """
        :param inner_group_num:
            Number of layers per layer group.
        :param num_hidden_groups:
            Number of hidden groups.
        """
        super(ALBERTLayerConfig, self).__init__(*args, **kwargs)
        self.inner_group_num = inner_group_num
        self.num_hidden_groups = num_hidden_groups


@dataclass
class ALBERTConfig(BERTConfig):
    """
    ALBERT (`Lan et al., 2022`_) model configuration.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    layer: ALBERTLayerConfig

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
        padding_id: int = 0,
    ):
        """
        :param embedding_width:
            Width of the embedding representations.
        :param hidden_width:
            Width of the transformer hidden layers.
        :param inner_group_num:
            Number of layers per layer group.
        :param intermediate_width:
            Width of the intermediate projection layer in the
            point-wise feed-forward layer.
        :param num_attention_heads:
            Number of self-attention heads.
        :param num_hidden_layers:
            Number of hidden layers.
        :param num_hidden_groups:
            Number of hidden groups.
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
        self.layer = ALBERTLayerConfig(
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
        self.padding_id = padding_id
