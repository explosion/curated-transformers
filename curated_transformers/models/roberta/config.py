from dataclasses import dataclass

import torch

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
        dtype: torch.dtype = torch.float32,
        layer_norm_eps=1e-05,
        n_positions=514,
        padding_id=1,
        n_types=1,
        n_pieces=50265,
        **kwargs
    ):
        """
        :param dtype:
            Data type to use for model parameters.
        :param embedding_width:
            Width of the embedding representations.
        :param hidden_width:
            Width of the transformer hidden layers.
        :param intermediate_width:
            Width of the intermediate projection layer in the
            point-wise feed-forward layer.
        :param n_attention_heads:
            Number of self-attention heads.
        :param n_hidden_layers:
            Number of hidden layers.
        :param attention_probs_dropout_prob:
            Dropout probabilty of the self-attention layers.
        :param hidden_dropout_prob:
            Dropout probabilty of the point-wise feed-forward and
            embedding layers.
        :param activation:
            Activation used by the pointwise feed-forward layers.
        :param n_pieces:
            Size of main vocabulary.
        :param n_types:
            Size of token type vocabulary.
        :param n_positions:
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
            n_positions=n_positions,
            n_types=n_types,
            n_pieces=n_pieces,
            **kwargs
        )
        self.dtype = dtype
        self.padding_id = padding_id
