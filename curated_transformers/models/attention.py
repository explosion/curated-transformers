import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .embeddings import QueryKeyRotaryEmbeddings
from .output import KeyValueCache


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


class QkvMode(IntEnum):
    """Modes of handling the query, key and value projections
    in the self-attention layer.
    """

    #: ``SEPARATE`` - Use separate projections for query, key and value.
    SEPARATE = (0,)

    #: ``MERGED_SPLIT_BEFORE`` - Use a merged projection for query, key and value, and split heads before splitting the query, key and value representations.
    MERGED_SPLIT_BEFORE = (1,)

    #: ``MERGED_SPLIT_AFTER`` - Use a merged projection for query, key and value, and split heads after splitting the query, key and value representations.
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
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
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

        model_dim = key.shape[-1]
        attn_scores = query @ key.transpose(-2, -1)
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
        attn_values = self.dropout(attn_weights @ value)

        return attn_values


@dataclass
class RotaryEmbeddingConfig:
    """
    Configuration for rotary embeddings.

    :param fraction: fraction of hidden width to apply rotary
        embeddings to. Must be in [0,1].
    :param base: Base in signifying the rotary embedding period.
    """

    fraction: float
    base: int = 10000


class SelfAttention(Module):
    """
    Transformer self-attention layer (Vaswani et al., 2017).
    """

    rotary_embeds: Optional[QueryKeyRotaryEmbeddings]

    def __init__(
        self,
        *,
        dropout_prob: float,
        hidden_width: int,
        num_attention_heads: int,
        qkv_mode: QkvMode,
        rotary_embeds: Optional[RotaryEmbeddingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """Construct a self-attention layer with rotary position embeddings.

        :param dropout_prob: Dropout to apply between the self-attention
            and output layers.
        :param hidden_width: Hidden width of the layer.
        :param num_attention_heads: Number of attention heads.
        :param qkv_mode: Handling mode for query, key and value.
        :param rotary_embeds: Configuration for rotary embeddings, rotary
            embeddings will not be used when set to ``None``.
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

        if rotary_embeds:
            self.rotary_embeds = QueryKeyRotaryEmbeddings(
                base=rotary_embeds.base,
                fraction=rotary_embeds.fraction,
                dims_per_head=self.dims_per_head,
            )
        else:
            self.rotary_embeds = None

        self.attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob,
        )

        if qkv_mode == QkvMode.SEPARATE:
            self.query = Linear(self.model_dim, self.model_dim, device=device)
            self.key = Linear(self.model_dim, self.model_dim, device=device)
            self.value = Linear(self.model_dim, self.model_dim, device=device)
        else:
            self.input = Linear(
                self.model_dim,
                self.model_dim * 3,
                device=device,
            )

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

        query, key, value = self._query_key_value(x)

        if self.rotary_embeds is not None:
            query, key = self.rotary_embeds(
                query=query, key=key, cache=cache, positions=positions
            )

        if cache is not None:
            cache_k = cache.key
            cache_v = cache.value

            key = torch.cat([cache_k, key], dim=-2)
            value = torch.cat([cache_v, value], dim=-2)

        attn = combine_heads(
            self.attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
            )
        )

        output = self.output(attn)

        if store_cache:
            return output, KeyValueCache(key=key, value=value)
        else:
            return output, None

    def _query_key_value(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute query, key and value representations for the input.

        :param x:
            Input
            **Shape:** (batch, seq_len, hidden_width)
        :returns:
            Query, key, value
            **Shape:** (batch, head, seq_len, width_per_head)
        """
        if self.qkv_mode == QkvMode.SEPARATE:
            query = self.query(x)
            key = self.key(x)
            value = self.value(x)

            query = split_heads(query, self.num_heads)
            key = split_heads(key, self.num_heads)
            value = split_heads(value, self.num_heads)
        elif self.qkv_mode == QkvMode.MERGED_SPLIT_BEFORE:
            proj = self.input(x)
            proj = split_heads(proj, self.num_heads)
            query, key, value = proj.chunk(3, dim=-1)
        else:
            proj = self.input(x)
            query, key, value = proj.split(
                [
                    self.num_heads * self.dims_per_head,
                    self.num_heads * self.dims_per_head,
                    self.num_heads * self.dims_per_head,
                ],
                dim=-1,
            )

            query = split_heads(query, self.num_heads)
            key = split_heads(key, self.num_heads)
            value = split_heads(value, self.num_heads)

        return query, key, value


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
