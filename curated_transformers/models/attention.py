from typing import Optional
import math
import torch
from torch import Tensor
from torch.nn import Linear, Module

from ..errors import Errors


class AttentionMask:
    bool_mask: Tensor
    _logit_mask: Optional[Tensor]

    def __init__(self, bool_mask: Tensor):
        if bool_mask.dtype != torch.bool:
            raise ValueError("Expected the attention mask to be of dtype 'torch.bool'")
        self.bool_mask = bool_mask
        self._logit_mask = torch.jit.annotate(Optional[Tensor], None)

    @property
    def logit_mask(self) -> Tensor:
        if self._logit_mask is None:
            # The value is `torch.finfo(attn_scores.dype).min`. Unfortunately,
            # we cannot use `torch.finfo` in TorchScript.
            self._logit_mask = (1.0 - self.bool_mask.int()) * -3.4028234663852886e38

        # Narrow type for TorchScript.
        logit_mask = self._logit_mask
        assert logit_mask is not None
        return logit_mask

    def dim(self) -> int:
        return self.bool_mask.dim()

    @property
    def shape(self):
        return self.bool_mask.shape


# https://www.tensorflow.org/text/tutorials/transformer#scaled_dot_product_attention
class ScaledDotProductAttention(Module):
    def __init__(self, *, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(
        self, k: Tensor, q: Tensor, v: Tensor, attn_mask: Optional[AttentionMask]
    ) -> Tensor:
        """
        Apply attention layer to the given key, query and value.

        Sequence elements that are marked with `False` in the attention mask
        are ignored by the attention mechanism (if a mask is provided).

        k (Tensor): key.
        q (Tensor): query.
        v (Tensor): value.
        attn_mask (Optional[AttentionMask]): attention mask.
        
        Shapes:
            k, q, v - (batch, heads, seq_len, width)
            attn_mask - (batch, seq_len)
        """

        if attn_mask is not None and attn_mask.dim() != 2:
            raise ValueError(
                "The attention mask must be a 2D-tensor of shape [batch, seq_len]"
            )

        model_dim = k.shape[-1]
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores /= math.sqrt(model_dim)

        if attn_mask is not None:
            # Replace tokens that we don't want to attend to with a large
            # negative value to zero them out during softmax normalization.
            batch, seq_len = attn_mask.shape
            attn_scores += attn_mask.logit_mask.view(batch, 1, 1, seq_len)

        attn_weights = attn_scores.softmax(dim=-1)
        attn_values = self.dropout(attn_weights @ v)

        return attn_values


# https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention
class SelfAttention(Module):
    """Transformer self-attention layer (Vaswani et al., 2017)."""

    def __init__(
        self,
        *,
        dropout_prob: float = 0.1,
        hidden_width: int = 768,
        num_attention_heads: int = 12,
    ):
        """Construct a self-attention layer.

        dropout_prob (float): dropout to apply between the self-attention
            and output layers (default: 0.1).
        hidden_width (int): hidden width of the layer (default: 768).
        num_attention_heads (int): number of attention heads (default: 12).
        """
        super().__init__()

        self.model_dim = hidden_width
        self.num_heads = num_attention_heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                Errors.E003.format(
                    hidden_width=self.model_dim, num_heads=self.num_heads
                )
            )

        self.dims_per_head = self.model_dim // self.num_heads
        self.attention = ScaledDotProductAttention(dropout_prob=dropout_prob)
        self.input = Linear(self.model_dim, self.model_dim * 3)
        self.output = Linear(self.model_dim, self.model_dim)

    def forward(self, x: Tensor, attn_mask: Optional[AttentionMask]) -> Tensor:
        """
        Apply self-attention layer to the input.

        Sequence elements that are marked with `False` in the attention mask
        are ignored by the attention mechanism (if a mask is provided).

        x (Tensor): input to apply self-attention to.
        attn_mask (Optional[AttentionMask]): attention mask.

        Shapes:
            x - (batch, seq_len, width)
            attn_mask - (batch, seq_len)
        """

        proj = self.input(x)
        q, k, v = proj.chunk(3, dim=-1)

        # (batch, head, seq_len, width_per_head)
        k = split_heads(k, self.num_heads)
        q = split_heads(q, self.num_heads)
        v = split_heads(v, self.num_heads)

        # (batch, seq_len, width)
        attn = combine_heads(self.attention(k, q, v, attn_mask))
        out = self.output(attn)

        return out


def combine_heads(x: Tensor) -> Tensor:
    """
    Combine the attention head representations. The inverse of
    'split_heads'.

    Shapes:
        x - (batch, head, seq_len, width_per_head)
        output - (batch, seq_len, width)
    """
    batch_size, head, seq_len, model_dim = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, head * model_dim)


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """
    Split the input by attention head. The caller must validate
    that the innermost dimension is divisable by the number of
    heads.

    x (Tensor): the tensor to split by head.
    num_heads (int): the number of attention heads.

    Shapes:
        x - (batch, seq_len, width)
        output - (batch, head, seq_len, width_per_head)
    """
    batch_size, seq_len, model_dim = x.size()

    assert model_dim % num_heads == 0

    dims_per_head = model_dim // num_heads

    return x.view(batch_size, seq_len, num_heads, dims_per_head).transpose(1, 2)
