import torch
from torch.nn import Linear, Module
from torch import Tensor

from .. import GeluNew
from ..attention import AttentionMask, SelfAttention
from .config import BertAttentionConfig, BertLayerConfig
from ...errors import Errors
from ..feedforward import PointwiseFeedForward


class BertFeedForward(Module):
    def __init__(self, config: BertLayerConfig):
        super().__init__()

        self.intermediate = Linear(config.hidden_width, config.intermediate_width)
        self.output = Linear(config.intermediate_width, config.hidden_width)
        if config.hidden_act == "relu":
            self.activation = torch.nn.ReLU()  # type: ignore
        elif config.hidden_act == "gelu":
            self.activation = torch.nn.GELU()  # type: ignore
        elif config.hidden_act == "gelu_new":
            # Ideally, we would use torch.nn.GELU(approximate="tanh"). However,
            # the differences between that and the manual Torch implementation
            # are large enough to fail tests comparing output to HF
            # transformers.
            self.activation = GeluNew()  # type: ignore
        else:
            raise ValueError(
                Errors.E004.format(activation_funcs=("relu", "gelu", "gelu_new"))
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, width)
        """
        out = self.intermediate(x)
        out = self.activation(out)
        out = self.output(out)
        return out


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
