import torch
from torch.nn import Module
from torch import Tensor

from .. import GeluNew
from ..attention import AttentionMask, ScaledDotProductAttention
from .config import BertAttentionConfig, BertLayerConfig


# https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention
class BertSelfAttention(Module):
    def __init__(self, config: BertAttentionConfig):
        super().__init__()

        self.model_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"model dimension '{self.model_dim}' not divisible by number of heads '{self.num_heads}'"
            )

        self.dims_per_head = self.model_dim // self.num_heads
        self.attention = ScaledDotProductAttention(dropout_prob=config.dropout_prob)
        self.input = torch.nn.Linear(self.model_dim, self.model_dim * 3)
        self.output = torch.nn.Linear(self.model_dim, self.model_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, emd_dim)
            output - (batch, head, seq_len, dims_per_head)
        """
        batch_size, seq_len, model_dim = x.size()
        return x.view(
            batch_size, seq_len, self.num_heads, self.dims_per_head
        ).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, head, seq_len, dims_per_head)
            output - (batch, seq_len, emd_dim)
        """
        batch_size, head, seq_len, model_dim = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_len, head * model_dim)
        )

    def forward(self, x: Tensor, attn_mask: AttentionMask) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
            attn_mask - (batch, seq_len)
        """

        proj = self.input(x)
        q, k, v = proj.chunk(3, dim=-1)

        # (batch, head, seq_len, dims_per_head)
        k = self._split_heads(k)
        q = self._split_heads(q)
        v = self._split_heads(v)

        # (batch, seq_len, model_dim)
        attn = self._combine_heads(self.attention(k, q, v, attn_mask))
        out = self.output(attn)

        return out


class BertFeedForward(Module):
    def __init__(self, config: BertLayerConfig):
        super().__init__()

        self.intermediate = torch.nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.output = torch.nn.Linear(config.intermediate_size, config.hidden_size)
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
            raise ValueError(f"unsupported activation function '{config.hidden_act}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
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
            layer_config.hidden_size, eps=layer_config.layer_norm_eps
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.ffn = BertFeedForward(layer_config)
        self.ffn_output_layernorm = torch.nn.LayerNorm(
            layer_config.hidden_size, eps=layer_config.layer_norm_eps
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)

    def forward(self, x: Tensor, attn_mask: AttentionMask) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
            attn_mask - (batch, seq_len)
        """
        attn_out = self.mha(x, attn_mask)
        attn_out = self.attn_output_dropout(attn_out)
        attn_out = self.attn_output_layernorm(x + attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)
        ffn_out = self.ffn_output_layernorm(attn_out + ffn_out)

        return ffn_out
