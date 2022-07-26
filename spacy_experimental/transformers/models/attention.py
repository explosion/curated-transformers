import math
from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

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

        `attn_mask` indicates elements to be masked with values of `1`
        """

        model_dim = k.shape[-1]
        qk = torch.mm(q, k.transpose(-2, -1))
        attn_scores = qk / math.sqrt(model_dim)

        # Replace masked-out elements with a large negative value
        # to zero them out during softmax normalization.
        if attn_mask:
            attn_scores = attn_scores.masked_fill(attn_mask, 1e-10)

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

        if attn_mask:
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
