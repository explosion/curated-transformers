from functools import partial
from typing import Any, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import LayerNorm

from ...layers.attention import AttentionHeads, QkvMode, SelfAttention
from ...layers.feedforward import PointwiseFeedForward
from ...layers.transformer import (
    EncoderLayer,
    TransformerDropouts,
    TransformerLayerNorms,
)
from ..hf_hub import FromHFHub
from ..transformer import TransformerEncoder
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import RoBERTaConfig
from .embeddings import RoBERTaEmbeddings

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="RoBERTaEncoder")


class RoBERTaEncoder(TransformerEncoder, FromHFHub):
    """
    RoBERTa (`Liu et al., 2019`_) encoder.

    .. _Liu et al., 2019: https://arxiv.org/abs/1907.11692
    """

    def __init__(self, config: RoBERTaConfig, *, device: Optional[torch.device] = None):
        """
        Construct a RoBERTa encoder.

        :param config:
            Encoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The encoder.
        """
        super().__init__()

        self.embeddings = RoBERTaEmbeddings(
            config.embedding, config.layer, padding_id=config.padding_id, device=device
        )
        self.max_seq_len = config.model_max_length

        hidden_width = config.layer.feedforward.hidden_width
        layer_norm = partial(
            LayerNorm,
            hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )
        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    attention_layer=SelfAttention(
                        attention_heads=AttentionHeads.uniform(
                            config.layer.attention.num_query_heads
                        ),
                        dropout_prob=config.layer.attention.dropout_prob,
                        hidden_width=hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=None,
                        use_bias=config.layer.attention.use_bias,
                        device=device,
                    ),
                    feed_forward_layer=PointwiseFeedForward(
                        activation=config.layer.feedforward.activation.module(),
                        hidden_width=hidden_width,
                        intermediate_width=config.layer.feedforward.intermediate_width,
                        use_bias=config.layer.feedforward.use_bias,
                        use_gate=config.layer.feedforward.use_gate,
                        device=device,
                    ),
                    dropouts=TransformerDropouts.layer_output_dropouts(
                        config.layer.dropout_prob
                    ),
                    layer_norms=TransformerLayerNorms(
                        attn_residual_layer_norm=layer_norm(),
                        ffn_residual_layer_norm=layer_norm(),
                    ),
                    parallel_attention=config.layer.attention.parallel_attention,
                )
                for _ in range(config.layer.num_hidden_layers)
            ]
        )

    @classmethod
    def convert_hf_state_dict(cls, params: Mapping[str, Tensor]):
        return convert_hf_state_dict(params)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config, device=device)
