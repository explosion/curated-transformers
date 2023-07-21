from dataclasses import dataclass

from ..config import (
    TransformerAttentionConfig,
    TransformerEmbeddingConfig,
    TransformerLayerConfig,
)


class BERTEmbeddingConfig(TransformerEmbeddingConfig):
    """
    BERT (`Devlin et al., 2018`_) embedding configuration.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    ...


class BERTAttentionConfig(TransformerAttentionConfig):
    """
    BERT (`Devlin et al., 2018`_) attention configuration.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    ...


class BERTLayerConfig(TransformerLayerConfig):
    """
    BERT (`Devlin et al., 2018`_) layer configuration.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    ...


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
            dropout_prob=attention_probs_dropout_prob,
            num_key_value_heads=num_attention_heads,
            num_query_heads=num_attention_heads,
            parallel_attention=False,
            rotary_embeddings=None,
            use_alibi=False,
            use_bias=True,
        )
        self.layer = BERTLayerConfig(
            hidden_width=hidden_width,
            intermediate_width=intermediate_width,
            num_hidden_layers=num_hidden_layers,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
            use_bias=True,
        )
        self.model_max_length = model_max_length
        self.padding_id = padding_id
