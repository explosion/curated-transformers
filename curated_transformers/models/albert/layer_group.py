from functools import partial
from typing import Optional

import torch
from torch import Tensor
from torch.nn import LayerNorm, Module, ModuleList

from ...layers.attention import AttentionHeads, AttentionMask, QkvMode, SelfAttention
from ...layers.feedforward import PointwiseFeedForward
from ...layers.transformer import (
    EncoderLayer,
    TransformerDropouts,
    TransformerLayerNorms,
)
from ..bert.config import BERTAttentionConfig
from .config import ALBERTLayerConfig


class ALBERTLayerGroup(Module):
    """
    ALBERT (`Lan et al., 2022`_) layer group.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    def __init__(
        self,
        layer_config: ALBERTLayerConfig,
        attention_config: BERTAttentionConfig,
        *,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        layer_norm = partial(
            LayerNorm,
            layer_config.hidden_width,
            layer_config.layer_norm_eps,
            device=device,
        )
        self.group_layers = ModuleList(
            [
                EncoderLayer(
                    attention_layer=SelfAttention(
                        attention_heads=AttentionHeads.uniform(
                            attention_config.num_attention_heads
                        ),
                        dropout_prob=attention_config.dropout_prob,
                        hidden_width=layer_config.hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=None,
                        use_bias=True,
                        device=device,
                    ),
                    feed_forward_layer=PointwiseFeedForward(
                        hidden_act=layer_config.hidden_act,
                        hidden_width=layer_config.hidden_width,
                        intermediate_width=layer_config.intermediate_width,
                        use_bias=True,
                        use_gate=False,
                        device=device,
                    ),
                    dropouts=TransformerDropouts.layer_output_dropouts(
                        layer_config.dropout_prob
                    ),
                    layer_norms=TransformerLayerNorms(
                        attn_residual_layer_norm=layer_norm(),
                        ffn_residual_layer_norm=layer_norm(),
                    ),
                    parallel_attention=False,
                )
                for _ in range(layer_config.inner_group_num)
            ]
        )

    def forward(self, input: Tensor, attention_mask: AttentionMask) -> Tensor:
        """
        Apply the ALBERT layer group to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer group to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        layer_output = input
        for layer in self.group_layers:
            layer_output = layer(layer_output, attention_mask)
        return layer_output
