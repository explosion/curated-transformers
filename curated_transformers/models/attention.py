import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Protocol, Tuple, TypeVar

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .embeddings import RotaryEmbeddings


class AttentionMask:
    bool_mask: Tensor

    def __init__(self, bool_mask: Tensor):
        if bool_mask.dtype != torch.bool:
            raise ValueError("Expected the attention mask to be of dtype 'torch.bool'")
        self.bool_mask = bool_mask

    def apply_logit_mask(self, input: Tensor) -> Tensor:
        """Use the attention mask to mask attention logits.

        :param input:
            Attention logits to apply the mask to. **Shape:**
            (batch, heads, query_len, key_len)
        :returns:
            Logits with the attention mask applied. **Shape:**
            (batch, heads, query_len, key_len)
        """
        blocked_value = torch.finfo(input.dtype).min
        batch, _, _, key_len = input.shape
        return torch.where(
            self.bool_mask.view(batch, 1, 1, key_len),
            input,
            blocked_value,
        )

    def dim(self) -> int:
        return self.bool_mask.dim()

    @property
    def shape(self):
        return self.bool_mask.shape


def apply_causal_mask(input: Tensor) -> Tensor:
    """Apply a causal mask.

    A causal mask ensures that tokens cannot attend to succeeding tokens.

    :param input:
        Attention logits to apply the causal mask to. **Shape:**
        (batch, heads, query_len, key_len)
    :returns:
        Logits with the causal mask applied. **Shape:**
        (batch, heads, query_len, key_len)
    """

    _, _, query_len, key_len = input.shape

    causal_mask = torch.tril(
        torch.full(
            (key_len, key_len),
            True,
            device=input.device,
        ),
    ).view(1, 1, key_len, key_len)
    causal_mask = causal_mask[:, :, key_len - query_len : key_len, :key_len]

    blocked_value = torch.finfo(input.dtype).min
    return torch.where(causal_mask, input, blocked_value)


CacheProtocolSelf = TypeVar("CacheProtocolSelf", bound="CacheProtocol")


class CacheProtocol(Protocol):
    def filter_batch_items(self: CacheProtocolSelf, mask: Tensor) -> CacheProtocolSelf:
        """
        Filter batch sequences from the cache.

        Sequences for which the mask is ``True`` are retained.

        :param mask:
            Mask of batch items to retain.
            **Shape:** (batch,)
        :returns:
            Filtered items.
        """
        ...


CacheT = TypeVar("CacheT", bound=CacheProtocol)


@dataclass
class KeyValueCache:
    """Cache type for layers that cache keys and values."""

    key: Tensor
    value: Tensor

    def filter_batch_items(self, mask: Tensor) -> "KeyValueCache":
        if mask.ndim != 1:
            raise ValueError(
                f"Cache mask must be a 1D tensor, has {mask.ndim} dimensions."
            )
        if mask.size(0) != self.key.size(0):
            raise ValueError(
                f"Cache mask size ({mask.size(0)}) must match cache batch size ({self.key.size(0)})."
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"Cache mask dtype must be bool, was: {mask.dtype}.")

        return KeyValueCache(key=self.key[mask], value=self.value[mask])


class QkvMode(IntEnum):
    """Modes of handling the query, key and value projections
    in the self-attention layer.

    - ``SEPARATE`` - Use separate projections for query, key and value.
    - ``MERGED_SPLIT_BEFORE`` - Use a merged projection for query, key and value, and split heads before splitting the query, key and value representations.
    - ``MERGED_SPLIT_AFTER`` - Use a merged projection for query, key and value, and split heads after splitting the query, key and value representations.
    """

    SEPARATE = (0,)
    MERGED_SPLIT_BEFORE = (1,)
    MERGED_SPLIT_AFTER = (2,)


# https://www.tensorflow.org/text/tutorials/transformer#scaled_dot_product_attention
class ScaledDotProductAttention(Module):
    """Scaled dot-product attention (Vaswani et al., 2017)"""

    def __init__(self, *, dropout_prob: float = 0.1):
        """
        :param dropout_prob: Dropout to apply to the final hidden
            representation.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(
        self,
        k: Tensor,
        q: Tensor,
        v: Tensor,
        attention_mask: Optional[AttentionMask],
        use_causal_mask: bool,
    ) -> Tensor:
        """
        Apply attention layer to the given key, query and value.

        Sequence elements that are marked with `False` in the attention mask
        are ignored by the attention mechanism (if a mask is provided).

        :param k: Key.
        :param q: Query.
        :param v: Value.
        :param attention_mask: Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            in attention.
        :param use_causal_mask: Mask out succeeding sequence elements when ``True``.
        :returns: hidden representation after applying attention.

        Shapes:
            k, q, v - (batch, heads, seq_len, width)
            attention_mask - (batch, seq_len)
            output - (batch, heads, seq_len, width)
        """

        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "The attention mask must be a 2D-tensor of shape [batch, seq_len]"
            )

        model_dim = k.shape[-1]
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores /= math.sqrt(model_dim)

        if use_causal_mask:
            # Replace tokens that occur after at token to zero them out
            # during softmax normalization.
            attn_scores = apply_causal_mask(attn_scores)

        if attention_mask is not None:
            # Replace tokens that we don't want to attend to with a large
            # negative value to zero them out during softmax normalization.
            attn_scores = attention_mask.apply_logit_mask(attn_scores)

        attn_weights = attn_scores.softmax(dim=-1)
        attn_values = self.dropout(attn_weights @ v)

        return attn_values


# https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention
class SelfAttention(Module):
    """Transformer self-attention layer (Vaswani et al., 2017)."""

    def __init__(
        self,
        *,
        dropout_prob: float,
        hidden_width: int,
        num_attention_heads: int,
        qkv_mode: QkvMode,
        device: Optional[torch.device] = None,
    ):
        """Construct a self-attention layer.

        :param dropout_prob: Dropout to apply between the self-attention
            and output layers.
        :param hidden_width: Hidden width of the layer.
        :param num_attention_heads: Number of attention heads.
        :param qkv_mode: Handling mode for query, key and value.
        :param device: Device on which the module is to be initialized.
        """
        super().__init__()

        self.model_dim = hidden_width
        self.num_heads = num_attention_heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"The hidden width of the transformer ({self.model_dim}) must be "
                f"divisible by the number of self-attention heads ({self.num_heads})"
            )

        self.dims_per_head = self.model_dim // self.num_heads
        self.qkv_mode = qkv_mode
        self.attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob,
        )

        if qkv_mode == QkvMode.SEPARATE:
            self.query = Linear(self.model_dim, self.model_dim, device=device)
            self.key = Linear(self.model_dim, self.model_dim, device=device)
            self.value = Linear(self.model_dim, self.model_dim, device=device)
        else:
            self.input = Linear(self.model_dim, self.model_dim * 3, device=device)
        self.output = Linear(self.model_dim, self.model_dim, device=device)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[AttentionMask],
        use_causal_mask: bool = False,
    ) -> Tensor:
        """
        Apply self-attention layer to the input.

        :param x: Input to apply self-attention to.
        :param attention_mask: Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            in attention.
        :param use_causal_mask: Mask out succeeding sequence elements when ``True``.
        :returns: Layer output.

        Shapes:
            x - (batch, seq_len, width)
            attention_mask - (batch, seq_len)
            output - (batch, seq_len, width)
        """

        if self.qkv_mode == QkvMode.SEPARATE:
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)

            q = split_heads(q, self.num_heads)
            k = split_heads(k, self.num_heads)
            v = split_heads(v, self.num_heads)
        elif self.qkv_mode == QkvMode.MERGED_SPLIT_BEFORE:
            # Project query, key, and value all at once and then split. We do
            # the projection this way to have the option of switching to fused
            # PyTorch kernels in the future.
            proj = self.input(x)
            proj = split_heads(proj, self.num_heads)
            q, k, v = proj.chunk(3, dim=-1)
        else:
            proj = self.input(x)
            q, k, v = proj.chunk(3, dim=-1)

            k = split_heads(k, self.num_heads)
            q = split_heads(q, self.num_heads)
            v = split_heads(v, self.num_heads)

        # (batch, seq_len, width)
        attn = combine_heads(self.attention(k, q, v, attention_mask, use_causal_mask))
        out = self.output(attn)

        return out


class SelfAttentionWithRotaryEmbeddings(Module):
    """Transformer self-attention layer with rotary position embeddings
    (Su et al., 2021).
    """

    def __init__(
        self,
        *,
        dropout_prob: float,
        hidden_width: int,
        num_attention_heads: int,
        qkv_mode: QkvMode,
        rotary_fraction: float,
        rotary_base: int = 10000,
        device: Optional[torch.device] = None,
    ):
        """Construct a self-attention layer with rotary position embeddings.

        :param dropout_prob: Dropout to apply between the self-attention
            and output layers.
        :param hidden_width: Hidden width of the layer.
        :param num_attention_heads: Number of attention heads.
        :param qkv_mode: Handling mode for query, key and value.
        :param rotary_fraction: fraction of hidden width to apply rotary
            embeddings to. Must be in [0,1].
        :param rotary_base: Base in signifying the rotary embedding period.
        :param device: Device on which the module is to be initialized.
        """

        super().__init__()

        self.model_dim = hidden_width
        self.num_heads = num_attention_heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"The hidden width of the transformer ({self.model_dim}) must be "
                f"divisible by the number of self-attention heads ({self.num_heads})"
            )

        self.dims_per_head = self.model_dim // self.num_heads
        self.qkv_mode = qkv_mode

        if not 0.0 <= rotary_fraction <= 1.0:
            raise ValueError(
                f"Rotary embedding fraction should be between 0.0 and 1.0 inclusive, was: {rotary_fraction}"
            )
        self.rotary_dims = int(rotary_fraction * self.dims_per_head)
        self.rotary_embeds = RotaryEmbeddings(
            width=self.rotary_dims, base=rotary_base, device=device
        )

        self.attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob,
        )
        if qkv_mode == QkvMode.SEPARATE:
            self.query = Linear(self.model_dim, self.model_dim, device=device)
            self.key = Linear(self.model_dim, self.model_dim, device=device)
            self.value = Linear(self.model_dim, self.model_dim, device=device)
        else:
            self.input = Linear(self.model_dim, self.model_dim * 3, device=device)
        self.output = Linear(self.model_dim, self.model_dim, device=device)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[AttentionMask],
        use_causal_mask: bool = False,
        cache: Optional[KeyValueCache] = None,
        store_cache: bool = False,
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply self-attention layer to the input.

        Sequence elements that are marked with `False` in the attention mask
        are ignored by the attention mechanism (if a mask is provided).

        :param x: Input to apply self-attention to.
        :param attention_mask: Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            in attention.
        :param use_causal_mask: Mask out succeeding sequence elements when ``True``.
        :param cache: Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param store_cache: Whether to cache the key/value representations for
            future reuse.
        :param positions: Input positions. Positions are needed to
            look up rotary embeddings. Normally, these positions are calculated
            automatically. But if the positions deviate for some reason, they
            can be provided through this argument.
        :returns: Layer output.

        Shapes:
            x - (batch, seq_len, width)
            attention_mask - (batch, seq_len)
            positions - (batch, seq_len)
        """

        if self.qkv_mode == QkvMode.SEPARATE:
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)

            q = split_heads(q, self.num_heads)
            k = split_heads(k, self.num_heads)
            v = split_heads(v, self.num_heads)
        elif self.qkv_mode == QkvMode.MERGED_SPLIT_BEFORE:
            proj = self.input(x)
            proj = split_heads(proj, self.num_heads)
            q, k, v = proj.chunk(3, dim=-1)
        else:
            proj = self.input(x)
            q, k, v = proj.chunk(3, dim=-1)

            k = split_heads(k, self.num_heads)
            q = split_heads(q, self.num_heads)
            v = split_heads(v, self.num_heads)

        dims_per_head = self.dims_per_head
        rotary_dims = self.rotary_dims

        if rotary_dims == dims_per_head:
            # Fast path: we apply rotary embeddings the full key/query vectors.
            k = self.rotary_embeds(k)
            q = self.rotary_embeds(q)
        else:
            # Otherwise, split up key/query vectors, apply rotary embeddings
            # and concatenate again.
            k_rotary, k_rest = k.split([rotary_dims, dims_per_head - rotary_dims], -1)
            q_rotary, q_rest = q.split([rotary_dims, dims_per_head - rotary_dims], -1)

            # If a cache was provided, but no positions, assume that the
            # positions of the current batch continue from the cache.
            if cache is not None and positions is None:
                cache_len = cache.key.size(-2)
                seq_len = k.size(-2)
                positions = torch.arange(
                    cache_len,
                    cache_len + seq_len,
                    dtype=torch.long,  # `torch.int32` isn't supported in indexing operations prior in torch<2.0.0.
                    device=k_rotary.device,
                ).repeat(x.size(0), 1)

            # Apply rotary embeddings.
            k_rotary = self.rotary_embeds(k_rotary, positions=positions)
            q_rotary = self.rotary_embeds(q_rotary, positions=positions)

            q = torch.cat([q_rotary, q_rest], dim=-1)
            k = torch.cat([k_rotary, k_rest], dim=-1)

        if cache is not None:
            cache_k = cache.key
            cache_v = cache.value

            k = torch.cat([cache_k, k], dim=-2)
            v = torch.cat([cache_v, v], dim=-2)

        attn = combine_heads(self.attention(k, q, v, attention_mask, use_causal_mask))

        output = self.output(attn)

        if store_cache:
            return output, KeyValueCache(key=k, value=v)
        else:
            return output, None


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
