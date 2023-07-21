from functools import partial
from typing import Any, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import LayerNorm

from ...layers.attention import AttentionHeads, AttentionMask, QkvMode, SelfAttention
from ...layers.feedforward import PointwiseFeedForward
from ...layers.transformer import (
    EncoderLayer,
    TransformerDropouts,
    TransformerLayerNorms,
)
from ..hf_hub import FromHFHub
from ..module import EncoderModule
from ..output import ModelOutput
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import RoBERTaConfig
from .embeddings import RoBERTaEmbeddings

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="RoBERTaEncoder")


class RoBERTaEncoder(EncoderModule, FromHFHub):
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
        self.padding_id = config.padding_id
        self.max_seq_len = config.model_max_length

        layer_norm = partial(
            LayerNorm,
            config.layer.hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )
        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    attention_layer=SelfAttention(
                        attention_heads=AttentionHeads.uniform(
                            config.attention.num_query_heads
                        ),
                        dropout_prob=config.attention.dropout_prob,
                        hidden_width=config.layer.hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=None,
                        use_bias=config.attention.use_bias,
                        device=device,
                    ),
                    feed_forward_layer=PointwiseFeedForward(
                        hidden_act=config.layer.hidden_act,
                        hidden_width=config.layer.hidden_width,
                        intermediate_width=config.layer.intermediate_width,
                        use_bias=config.layer.use_bias,
                        use_gate=False,
                        device=device,
                    ),
                    dropouts=TransformerDropouts.layer_output_dropouts(
                        config.layer.dropout_prob
                    ),
                    layer_norms=TransformerLayerNorms(
                        attn_residual_layer_norm=layer_norm(),
                        ffn_residual_layer_norm=layer_norm(),
                    ),
                    parallel_attention=config.attention.parallel_attention,
                )
                for _ in range(config.layer.num_hidden_layers)
            ]
        )

    def _create_attention_mask(self, x: Tensor) -> AttentionMask:
        return AttentionMask(x.ne(self.padding_id))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        if attention_mask is None:
            attention_mask = self._create_attention_mask(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layer_outputs = []
        for layer in self.layers:
            layer_output, _ = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)

        return ModelOutput(
            embedding_output=embeddings, layer_hidden_states=layer_outputs
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
