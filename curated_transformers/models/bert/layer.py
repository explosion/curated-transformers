import torch
from torch.nn import Linear, Module
from torch import Tensor

from ..attention import AttentionMask, SelfAttention
from .config import BertAttentionConfig, BertLayerConfig
from ..feedforward import PointwiseFeedForward


class BertEncoderLayer(Module):
    def __init__(
        self, layer_config: BertLayerConfig, attention_config: BertAttentionConfig
    ):
        super().__init__()

        self.mha = SelfAttention(
            dropout_prob=attention_config.dropout_prob,
            hidden_width=attention_config.hidden_width,
            num_attention_heads=attention_config.num_attention_heads,
        )
        self.attn_output_layernorm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.ffn = PointwiseFeedForward(
            hidden_act=layer_config.hidden_act,
            hidden_width=layer_config.hidden_width,
            intermediate_width=layer_config.intermediate_width,
        )
        self.ffn_output_layernorm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)

    def forward(self, x: Tensor, attn_mask: AttentionMask) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, width)
            attn_mask - (batch, seq_len)
        """
        attn_out = self.mha(x, attn_mask)
        attn_out = self.attn_output_dropout(attn_out)
        attn_out = self.attn_output_layernorm(x + attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)
        ffn_out = self.ffn_output_layernorm(attn_out + ffn_out)

        return ffn_out
