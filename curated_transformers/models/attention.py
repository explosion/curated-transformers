from typing import Optional
import math
import torch
from torch import Tensor
from torch.nn import Module

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
