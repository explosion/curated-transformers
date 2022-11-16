from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from torch.nn import Module


@dataclass
class AttentionMask:
    bool_mask: Tensor
    _logit_mask: Optional[Tensor] = None

    def __post_init__(self):
        if self.bool_mask.dtype != torch.bool:
            raise ValueError(
                f"attention mask of dtype torch.bool expected, was {self.bool_mask.dtype}"
            )

    @property
    def logit_mask(self) -> Tensor:
        if self._logit_mask is None:
            # The value is `torch.finfo(attn_scores.dype).min`. Unfortunately,
            # we cannot use `torch.finfo` in TorchScript.
            self._logit_mask = (1.0 - self.bool_mask.int()) * -3.4028234663852886e38
        return self._logit_mask

    def dim(self) -> int:
        return self.bool_mask.dim()

    @property
    def shape(self) -> Tuple:
        return self.bool_mask.shape


# https://www.tensorflow.org/text/tutorials/transformer#scaled_dot_product_attention
class ScaledDotProductAttention(Module):
    def __init__(self, *, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(
        self, k: Tensor, q: Tensor, v: Tensor, attn_mask: AttentionMask
    ) -> Tensor:
        """
        Shapes:
            k, q, v, attn_mask - (batch, heads, seq, model_dim)
        """

        if attn_mask.dim() != 2:
            raise ValueError(
                f"attention mask dim mismatch, expected '2' but received {attn_mask.dim()}"
            )

        model_dim = k.shape[-1]
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores /= math.sqrt(model_dim)

        # Replace tokens that we don't want to attend to with a large
        # negative value to zero them out during softmax normalization.
        batch, seq_len = attn_mask.shape
        attn_scores += attn_mask.logit_mask.view(batch, 1, 1, seq_len)

        attn_weights = attn_scores.softmax(dim=-1)
        attn_values = self.dropout(attn_weights @ v)

        return attn_values
