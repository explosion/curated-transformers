import torch
from torch.nn import Module
from torch import Tensor

from .. import GeluNew
from ..attention import AttentionMask, ScaledDotProductAttention
from .config import BertAttentionConfig, BertLayerConfig
from ..linear import Linear
from ...._compat import has_torch_sdp_attention
from ....errors import Errors


# https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention
class BertSelfAttention(Module):
    def __init__(self, config: BertAttentionConfig):
        super().__init__()

        self.dropout_prob = config.dropout_prob
        self.model_dim = config.hidden_width
        self.num_heads = config.num_attention_heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                Errors.E003.format(
                    hidden_width=self.model_dim, num_heads=self.num_heads
                )
            )

        self.dims_per_head = self.model_dim // self.num_heads
        self.attention = ScaledDotProductAttention(dropout_prob=config.dropout_prob)
        self.input = Linear(self.model_dim, self.model_dim * 3)
        self.output = Linear(self.model_dim, self.model_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, width)
            output - (batch, head, seq_len, width_per_head)
        """
        batch_size, seq_len, model_dim = x.size()
        return x.view(
            batch_size, seq_len, self.num_heads, self.dims_per_head
        ).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, head, seq_len, width_per_head)
            output - (batch, seq_len, width)
        """
        batch_size, head, seq_len, model_dim = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_len, head * model_dim)
        )

    def forward(self, x: Tensor, attn_mask: AttentionMask) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, width)
            attn_mask - (batch, seq_len)
        """
        proj = self.input(x)
        q, k, v = proj.chunk(3, dim=-1)

        # (batch, head, seq_len, width_per_head)
        k = self._split_heads(k)
        q = self._split_heads(q)
        v = self._split_heads(v)

        # attn: (batch, head, seq_len, with_per_head)
        if has_torch_sdp_attention:
            batch, seq_len = attn_mask.shape
            mask = attn_mask.bool_mask.view(batch, 1, 1, seq_len)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, mask, self.dropout_prob if self.training else 0.0
            )
        else:
            attn = self.attention(k, q, v, attn_mask)

        # attn: (batch, seq_len, width)
        attn = self._combine_heads(attn)

        out = self.output(attn)

        return out


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

        self.mha = BertSelfAttention(attention_config)
        self.attn_output_layernorm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.ffn = BertFeedForward(layer_config)
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
