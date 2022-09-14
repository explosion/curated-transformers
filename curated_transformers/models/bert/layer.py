import math
from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

from ..attention import ScaledDotProductAttention

# https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention
class BertSelfAttention(Module):
    def __init__(self, model_dim: int, n_heads: int, *, dropout: float = 0.1):
        super().__init__()

        if model_dim % n_heads != 0:
            raise ValueError(
                f"model dimension '{model_dim}' not divisible by number of heads '{n_heads}'"
            )

        self.model_dim = model_dim
        self.num_heads = n_heads
        self.dims_per_head = model_dim // n_heads
        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.query = torch.nn.Linear(model_dim, model_dim)
        self.key = torch.nn.Linear(model_dim, model_dim)
        self.value = torch.nn.Linear(model_dim, model_dim)

        self.output = torch.nn.Linear(model_dim, model_dim)

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

    def forward(
        self, k: Tensor, q: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Shapes:
            k, q, v - (batch, seq_len, model_dim)
            attn_mask - (batch, seq_len)

        `attn_mask` indicates elements to be masked with values of `1`
        """

        k = self.key(k)
        q = self.query(q)
        v = self.value(v)

        # (batch, head, seq_len, dims_per_head)
        k = self._split_heads(k)
        q = self._split_heads(q)
        v = self._split_heads(v)

        if attn_mask is not None:
            if attn_mask.dim() != 2:
                raise ValueError(
                    f"attention mask dim mismatch, expected '2' but received {attn_mask.dim()}"
                )
            batch, seq_len = attn_mask.size()
            attn_mask = attn_mask.contiguous().view(batch, 1, 1, seq_len)

        # (batch, seq_len, model_dim)
        attn = self._combine_heads(self.attention(k, q, v, attn_mask))
        out = self.output(attn)

        return out


class BertFeedForward(Module):
    def __init__(
        self,
        hidden_dim: int,
        model_dim: int,
        *,
        activation: str = "relu",
    ):
        super().__init__()

        self.intermediate = torch.nn.Linear(model_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, model_dim)
        if activation == "relu":
            self.activation = torch.nn.ReLU()  # type: ignore
        elif activation == "gelu":
            self.activation = torch.nn.GELU()  # type: ignore
        else:
            raise ValueError(f"unsupported activation function '{activation}")

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
        self,
        model_dim: int,
        ffn_dim: int,
        num_attn_heads: int,
        *,
        activation: str = "relu",
        attn_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.mha = BertSelfAttention(
            model_dim, n_heads=num_attn_heads, dropout=attn_dropout
        )
        self.attn_output_layernorm = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.attn_output_dropout = torch.nn.Dropout(p=hidden_dropout)

        self.ffn = BertFeedForward(
            hidden_dim=ffn_dim,
            model_dim=model_dim,
            activation=activation,
        )
        self.ffn_output_layernorm = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.ffn_output_dropout = torch.nn.Dropout(p=hidden_dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
            mask - (batch, seq_len)

        `mask` indicates elements to be masked with values of `1`
        """
        attn_out = self.mha(x, x, x, mask)
        attn_out = self.attn_output_dropout(attn_out)
        attn_out = self.attn_output_layernorm(x + attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)
        ffn_out = self.ffn_output_layernorm(attn_out + ffn_out)

        return ffn_out
