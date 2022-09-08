import math
from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinusoidalPositionalEmbedding(Module):
    def __init__(self, dim: int, max_len: int, *, normalize=True):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if normalize == True:
            l2 = torch.norm(pe, dim=-1)
            pe /= l2.unsqueeze(-1)

        self.pe = pe
        self.pe.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len)
        """
        return self.pe[x.size(1), :]


# https://www.tensorflow.org/text/tutorials/transformer#scaled_dot_product_attention
class ScaledDotProductAttention(Module):
    def __init__(self, *, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self, k: Tensor, q: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Shapes:
            k, q, v, attn_mask - (batch, heads, seq, model_dim)

        `attn_mask` indicates elements to attend to with `1` (and `0` otherwise)
        """

        model_dim = k.shape[-1]
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores /= math.sqrt(model_dim)

        # Replace tokens that we don't want to attend to with a large
        # negative value to zero them out during softmax normalization.
        if attn_mask is not None:
            attn_scores += (1.0 - attn_mask) * torch.finfo(attn_scores.dtype).min

        attn_weights = attn_scores.softmax(dim=-1)
        attn_values = self.dropout(attn_weights @ v)

        return attn_values


# https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention
class MultiHeadAttention(Module):
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


class PointwiseFeedForwardLayer(Module):
    def __init__(
        self,
        hidden_dim: int,
        model_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.1,
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

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
        """
        out = self.intermediate(x)
        out = self.activation(out)
        out = self.output(out)
        out = self.dropout(out)
        return out


class EncoderLayer(Module):
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

        self.mha = MultiHeadAttention(
            model_dim, n_heads=num_attn_heads, dropout=attn_dropout
        )
        self.attn_output_layernorm = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.attn_output_dropout = torch.nn.Dropout(p=hidden_dropout)

        self.ffn = PointwiseFeedForwardLayer(
            hidden_dim=ffn_dim,
            model_dim=model_dim,
            activation=activation,
            dropout=hidden_dropout,
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
        attn_out = self.attn_output_layernorm(x + attn_out)
        attn_out = self.attn_output_dropout(attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_output_layernorm(attn_out + ffn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)

        return ffn_out
