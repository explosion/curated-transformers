from dataclasses import dataclass

from ..bert import BERTConfig


@dataclass
class RoBERTaConfig(BERTConfig):
    """
    RoBERTa (`Liu et al., 2019`_) model configuration.

    .. _Liu et al., 2019: https://arxiv.org/abs/1907.11692
    """

    def __init__(
        self,
        *args,
        layer_norm_eps=1e-05,
        max_position_embeddings=514,
        padding_id=1,
        type_vocab_size=1,
        vocab_size=50265,
        **kwargs
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
        :param padding_id:
            Index of the padding meta-token.
        """
        super(RoBERTaConfig, self).__init__(
            *args,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size,
            **kwargs
        )

        self.padding_id = padding_id
