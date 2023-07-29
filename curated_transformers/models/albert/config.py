from dataclasses import dataclass

from ...layers.activations import Activation
from ..config import (
    TransformerAttentionLayerConfig,
    TransformerEmbeddingLayerConfig,
    TransformerFeedForwardLayerConfig,
    TransformerLayerConfig,
)


@dataclass
class ALBERTLayerConfig(TransformerLayerConfig):
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
class ALBERTConfig:
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
        activation: Activation = Activation.GELUNew,
        vocab_size: int = 30000,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        model_max_length: int = 512,
        layer_norm_eps: float = 1e-12,
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
        :param activation:
            Activation used by the pointwise feed-forward layers.
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
        """
        self.embedding = TransformerEmbeddingLayerConfig(
            embedding_width=embedding_width,
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )
        self.layer = ALBERTLayerConfig(
            attention=TransformerAttentionLayerConfig(
                hidden_width=hidden_width,
                dropout_prob=attention_probs_dropout_prob,
                num_key_value_heads=num_attention_heads,
                num_query_heads=num_attention_heads,
                parallel_attention=False,
                rotary_embeddings=None,
                use_alibi=False,
                use_bias=True,
            ),
            feedforward=TransformerFeedForwardLayerConfig(
                hidden_width=hidden_width,
                intermediate_width=intermediate_width,
                activation=activation,
                use_bias=True,
                use_gate=False,
            ),
            num_hidden_layers=num_hidden_layers,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
            inner_group_num=inner_group_num,
            num_hidden_groups=num_hidden_groups,
        )
        self.model_max_length = model_max_length
