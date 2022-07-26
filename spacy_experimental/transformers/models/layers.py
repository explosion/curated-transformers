from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

from .attention import MultiHeadAttention


class PointwiseFeedForwardLayer(Module):
    def __init__(self, hidden_dim: int, model_dim: int, *, activation: str = "relu", dropout: float = 0.1):
        super().__init__()

        self.intermediate = torch.nn.Linear(model_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, model_dim)
        if activation == "relu":
            self.activation = torch.nn.ReLU() # type: ignore
        elif activation == "gelu":
            self.activation = torch.nn.GELU() # type: ignore
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
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            model_dim, n_heads=num_attn_heads, dropout=attn_dropout
        )
        self.attn_output_layernorm = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.attn_output_dropout = torch.nn.Dropout(p=hidden_dropout)

        self.ffn = PointwiseFeedForwardLayer(
            hidden_dim=ffn_dim, model_dim=model_dim, activation=activation, dropout=hidden_dropout
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
