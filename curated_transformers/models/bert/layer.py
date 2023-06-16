from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from ..attention import AttentionMask, QkvHeadSharing, QkvMode, SelfAttention
from ..feedforward import PointwiseFeedForward
from .config import BertAttentionConfig, BertLayerConfig


class BertEncoderLayer(Module):
    def __init__(
        self,
        layer_config: BertLayerConfig,
        attention_config: BertAttentionConfig,
        *,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.mha = SelfAttention(
            dropout_prob=attention_config.dropout_prob,
            hidden_width=attention_config.hidden_width,
            qkv_head_sharing=QkvHeadSharing.NONE,
            num_attention_heads=attention_config.num_attention_heads,
            qkv_mode=QkvMode.SEPARATE,
            rotary_embeds=None,
            use_bias=True,
            device=device,
        )
        self.attn_output_layernorm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps, device=device
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.ffn = PointwiseFeedForward(
            hidden_act=layer_config.hidden_act,
            hidden_width=layer_config.hidden_width,
            intermediate_width=layer_config.intermediate_width,
            use_bias=True,
            device=device,
        )
        self.ffn_output_layernorm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps, device=device
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)

    def forward(self, x: Tensor, attention_mask: AttentionMask) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, width)
            attention_mask - (batch, seq_len)
        """
        attn_out, _ = self.mha(x, attention_mask)
        attn_out = self.attn_output_dropout(attn_out)
        attn_out = self.attn_output_layernorm(x + attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)
        ffn_out = self.ffn_output_layernorm(attn_out + ffn_out)

        return ffn_out
