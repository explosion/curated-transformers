import math
from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntEnum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module

from .cache import KeyValueCache
from .embeddings import QueryKeyRotaryEmbeddings

_TORCH_SDP: ContextVar[bool] = ContextVar("torch_sdp", default=False)


@contextmanager
def enable_torch_sdp(use_torch_sdp: bool = True):
    """
    Enables Torch scaled dot product attention.

    Torch provides an implementation of scaled dot product attention that
    has many optimizations. For instance, in some scenarios Flash Attention
    is applied (`Dao et al., 2022`_). We do not use the Torch implementation
    by default, because it is still in beta.

    This context manager enables use of the Torch implementation of scaled
    dot product attention.

    .. code-block:: python

        with enable_torch_sdp():
            Y = bert_encoder(X)

    .. _Dao et al., 2022:  https://arxiv.org/abs/2205.14135
    """
    token = _TORCH_SDP.set(use_torch_sdp)
    try:
        yield
    finally:
        _TORCH_SDP.reset(token)


class AttentionMask:
    """
    Mask for attention calculation. Sequence elements for which the
    corresponding mask element is set to ``False`` are ignored during
    attention calculation.
    """

    bool_mask: Tensor
    _logit_mask: Optional[Tensor]

    def __init__(self, bool_mask: Tensor):
        """
        Construct an attention mask.

        :param bool_mask:
            The boolean mask.
        """
        if bool_mask.dtype != torch.bool:
            raise ValueError("Expected the attention mask to be of dtype 'torch.bool'")

        if bool_mask.dim() == 2:
            batch_len, key_len = bool_mask.shape
            self.bool_mask = bool_mask.view([batch_len, 1, 1, key_len])
        elif bool_mask.dim() == 4:
            self.bool_mask = bool_mask
        else:
            raise ValueError(
                "The attention mask must be a tensor of shape [batch, key_len] or "
                "[batch_len, heads, query_len, key_len]"
            )

        self._logit_mask = None

    def apply_logit_mask(self, input: Tensor) -> Tensor:
        """
        Use the attention mask to mask attention logits.

        :param input:
            Attention logits to apply the mask to.

            *Shape:* ``(batch_size, heads, query_len, key_len)``
        :returns:
            Logits with the attention mask applied.

            *Shape:* ``(batch_size, heads, query_len, key_len)``
        """
        blocked_value = torch.finfo(input.dtype).min
        return torch.where(self.bool_mask, input, blocked_value)

    def dim(self) -> int:
        return self.bool_mask.dim()

    def merge_mask(self, other: "AttentionMask") -> "AttentionMask":
        return AttentionMask(self.bool_mask.logical_and(other.bool_mask))

    def logit_mask(self, dtype: torch.dtype):
        if self._logit_mask is None:
            self._logit_mask = (1.0 - self.bool_mask.to(dtype)) * torch.finfo(dtype).min
        return self._logit_mask

    @property
    def shape(self):
        return self.bool_mask.shape


def create_causal_mask(query: Tensor, key: Tensor) -> AttentionMask:
    """
    Create a causal mask. A causal mask ensures that tokens
    cannot attend to succeeding tokens.

    :param query:
        Query to compute the causal mask for.

        *Shape:* ``(batch_size, heads, query_len, head_dim)``
    :param key:
        Key to compute the causal mask for.

        *Shape:* ``(batch_size, heads, key_len, head_dim)``
    :returns:
        The causal mask.

        *Shape:* ``(batch_size, heads, query_len, key_len)``
    """
    query_len = query.size(2)
    key_len = key.size(2)

    causal_mask = torch.tril(
        torch.full(
            (key_len, key_len),
            True,
            device=query.device,
        ),
    ).view(1, 1, key_len, key_len)
    return AttentionMask(causal_mask[:, :, key_len - query_len : key_len, :key_len])


class AttentionHeads:
    def __init__(self, *, num_query_heads: int, num_key_value_heads: int):
        """
        Construct an attention head configuration. This constructor must
        not be used directly, its signature may change even within a semver
        version. Use the factory methods instead.

        :param num_query_heads:
            Number of query heads.
        :param num_key_value_heads:
            Number of key/value heads.

        :meta private:
        """
        self._num_query_heads = num_query_heads
        self._num_key_value_heads = num_key_value_heads

    @classmethod
    def uniform(cls, num_attention_heads: int) -> "AttentionHeads":
        """
        Construct a head configuration where query, key, and value have the
        same number of attention heads.

        :param num_attention_heads:
            Number of attention heads.
        """
        return cls(
            num_query_heads=num_attention_heads, num_key_value_heads=num_attention_heads
        )

    @classmethod
    def multi_query(cls, num_query_heads: int) -> "AttentionHeads":
        """
        Construct a multi-query attention configuration: key shares heads,
        value shares heads, query has separate heads (`Shazeer et al., 2019`_).

        .. _Shazeer et al., 2019: https://arxiv.org/abs/1911.02150

        :param num_query_heads:
            Number of query heads.
        """
        return cls(num_query_heads=num_query_heads, num_key_value_heads=1)

    @classmethod
    def key_value_broadcast(
        cls, *, num_query_heads: int, num_key_value_heads: int
    ) -> "AttentionHeads":
        """
        Construct a head configuration where query has a larger number
        of heads than key and value. Key/value heads are broadcast to
        correspond to the number of query heads.

        :param num_query_heads:
            Number of attention heads. Must be a multiple of
            ``num_key_value_heads``.
        :param num_key_value_heads:
            Number of key and value heads.
        """

        if (
            num_query_heads < num_key_value_heads
            or num_query_heads % num_key_value_heads != 0
        ):
            raise ValueError(
                f"Number of query heads ({num_query_heads}) must be a multiple of key/value heads ({num_key_value_heads})"
            )

        return cls(
            num_query_heads=num_query_heads, num_key_value_heads=num_key_value_heads
        )


class QkvMode(IntEnum):
    """
    How the query, key and value projections are handled in
    the self-attention layer.
    """

    #: ``SEPARATE`` - Use separate projections for query, key and value.
    SEPARATE = (0,)

    #: ``MERGED_SPLIT_BEFORE`` - Use a merged projection for query,
    #: key and value, and split heads before splitting the query, key
    #: and value representations. This ordering is incompatible with
    #: head sharing in keys or values.

    MERGED_SPLIT_BEFORE = (1,)

    #: ``MERGED_SPLIT_AFTER`` - Use a merged projection for query,
    #: key and value, and split heads after splitting the query, key
    #: and value representations.
    MERGED_SPLIT_AFTER = (2,)


# https://www.tensorflow.org/text/tutorials/transformer#scaled_dot_product_attention
class ScaledDotProductAttention(Module):
    """
    Scaled dot-product attention (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, *, dropout_prob: float = 0.1):
        """
        Construct a scaled dot-product attention module.

        :param dropout_prob:
            Dropout to apply to the final hidden representation.
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
    ) -> Tensor:
        """
        Apply attention layer to the given key, query and value.

        Sequence elements that are marked with `False` in the attention mask
        are ignored by the attention mechanism (if a mask is provided).

        :param k:
            Key.

            *Shape:* ``(batch_size, heads, seq_len, width)``
        :param q:
            Query.

            *Shape:* ``(batch_size, heads, seq_len, width)``
        :param v:
            Value.

            *Shape:* ``(batch_size, heads, seq_len, width)``
        :param attention_mask:

            Attention mask. Sequence elements for which the corresponding mask
            element is set to ``False`` are ignored in attention.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Attention values.

            *Shape:* ``(batch_size, heads, seq_len, width)``
        """
        model_dim = key.shape[-1]
        attn_scores = query @ key.transpose(-2, -1)
        attn_scores /= math.sqrt(model_dim)

        if attention_mask is not None:
            # Replace tokens that we don't want to attend to with a large
            # negative value to zero them out during softmax normalization.
            attn_scores = attention_mask.apply_logit_mask(attn_scores)

        attn_weights = attn_scores.softmax(dim=-1)
        attn_values = self.dropout(attn_weights @ value)

        return attn_values


class SelfAttention(Module):
    """
    Transformer self-attention layer (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    rotary_embeds: Optional[QueryKeyRotaryEmbeddings]

    def __init__(
        self,
        *,
        attention_heads: AttentionHeads,
        dropout_prob: float,
        hidden_width: int,
        qkv_mode: QkvMode,
        rotary_embeds: Optional[QueryKeyRotaryEmbeddings] = None,
        use_bias: bool,
        device: Optional[torch.device] = None,
    ):
        """
        Construct a self-attention layer with rotary position embeddings.

        :param attention_heads:
            Attention head configuration.
        :param dropout_prob:
            Dropout to apply between the self-attention and output layers.
        :param hidden_width:
            Hidden width of the layer.
        :param qkv_mode:
            Handling mode for query, key and value.
        :param rotary_embeds:
            Rotary embeddings. Rotary embeddings will not be used when set
            to ``None``.
        :param use_bias:
            Use biases for linear layers.
        :param device:
            Device on which the module is to be initialized.
        """

        super().__init__()

        self.dropout_prob = dropout_prob
        self.attention_heads = attention_heads
        if hidden_width % attention_heads._num_query_heads != 0:
            raise ValueError(
                f"The hidden width of the transformer ({hidden_width}) must be "
                f"divisible by the number of query heads ({attention_heads._num_query_heads})"
            )

        self.dims_per_head = hidden_width // attention_heads._num_query_heads
        self.qkv_mode = qkv_mode

        self.rotary_embeds = rotary_embeds

        self.attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob,
        )

        if (
            qkv_mode == QkvMode.MERGED_SPLIT_BEFORE
            and attention_heads._num_query_heads != attention_heads._num_key_value_heads
        ):
            raise ValueError(
                "QkvMode.MERGED_SPLIT_BEFORE is incompatible with key/value head sharing"
            )

        key_value_dim = attention_heads._num_key_value_heads * self.dims_per_head
        if qkv_mode == QkvMode.SEPARATE:
            self.query = Linear(
                hidden_width,
                hidden_width,
                bias=use_bias,
                device=device,
            )
            self.key = Linear(
                hidden_width,
                key_value_dim,
                bias=use_bias,
                device=device,
            )
            self.value = Linear(
                hidden_width,
                key_value_dim,
                bias=use_bias,
                device=device,
            )
        else:
            self.input = Linear(
                hidden_width,
                hidden_width + 2 * key_value_dim,
                bias=use_bias,
                device=device,
            )

        self.output = Linear(hidden_width, hidden_width, bias=use_bias, device=device)

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[AttentionMask],
        use_causal_mask: bool = False,
        cache: Optional[KeyValueCache] = None,
        store_cache: bool = False,
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply self-attention layer to the input.

        :param input:
            Input to apply self-attention to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            in attention.

            *Shape:* ``(batch_size, seq_len)``
        :param use_causal_mask:
            Mask out succeeding sequence elements when ``True``.
        :param cache:
            Key/value cache to avoid recomputing key/value representations
            for tokens that were previously seen.
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :param positions:
            Input positions. Positions are needed to look up rotary embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this
            argument.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """

        query, key, value = self._query_key_value(input)

        if self.rotary_embeds is not None:
            query, key = self.rotary_embeds(
                query=query, key=key, cache=cache, positions=positions
            )

        if cache is not None:
            cache_k = cache.key
            cache_v = cache.value

            key = torch.cat([cache_k, key], dim=-2)
            value = torch.cat([cache_v, value], dim=-2)

        combined_mask = attention_mask
        if use_causal_mask:
            causal_mask = create_causal_mask(query, key)
            combined_mask = (
                causal_mask
                if combined_mask is None
                else combined_mask.merge_mask(causal_mask)
            )

        if _TORCH_SDP.get():
            # We can't pass a bool mask, because it is currently broken:
            # https://github.com/pytorch/pytorch/issues/103749
            attn = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None
                if combined_mask is None
                else combined_mask.logit_mask(query.dtype),
                dropout_p=self.dropout_prob if self.training else 0.0,
            )
        else:
            attn = self.attention(
                query=query,
                key=key,
                value=value,
                attention_mask=combined_mask,
            )

        attn = combine_heads(attn)

        output = self.output(attn)

        if store_cache:
            return output, KeyValueCache(key=key, value=value)
        else:
            return output, None

    def _query_key_value(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute query, key and value representations for the input.

        :param input:
            Input

            *Shape:* ``(batch_size, seq_len, hidden_width)``
        :returns:
            Query, key, value

            *Shape:* ``(batch_size, head, seq_len, width_per_head)``
        """

        num_key_value_heads = self.attention_heads._num_key_value_heads
        num_query_heads = self.attention_heads._num_query_heads
        if self.qkv_mode == QkvMode.SEPARATE:
            query = self.query(input)
            key = self.key(input)
            value = self.value(input)

            query = split_heads(query, num_query_heads)
            key = split_heads(key, num_key_value_heads)
            value = split_heads(value, num_key_value_heads)
        elif self.qkv_mode == QkvMode.MERGED_SPLIT_BEFORE:
            proj = self.input(input)
			# Same number of heads for query, key and value
			# since we cannot share heads in this mode.
            proj = split_heads(proj, num_query_heads) 
            query, key, value = proj.chunk(3, dim=-1)
        else:
            proj = self.input(input)

            # This splitting method is used by Falcon. There are other
            # (simpler) splitting methods. For instance, we could group all
            # the heads together and then split them. In the future we
            # could split conditional on a field in AttentionHeads.
            query, key, value = split_heads_grouped_qkv(
                proj, self.attention_heads, self.dims_per_head
            )

        if num_key_value_heads != 1 and num_key_value_heads != num_query_heads:
            # If all the key/value representations are shared, their head
            # dimension is 1 and the heads will broadcast in later ops.
            # However, we have multiple heads, we have to broadcast the
            # heads explicitly. Like splitting, there are multiple ways
            # to do this. The current interleaving repeat is used for
            # compatibility with falcon.
            n_tiling = num_query_heads // num_key_value_heads
            key = key.repeat_interleave(n_tiling, dim=1)
            value = value.repeat_interleave(n_tiling, dim=1)

        return query, key, value


def split_heads(input: Tensor, num_heads: int) -> Tensor:
    """
    Split the input by attention head. The caller must validate
    that the innermost dimension is divisable by the number of
    heads.

    :param input:
        Tensor to split by head.

        *Shape:* ``(batch_size, seq_len, hidden_width)``
    :param num_heads:
        Number of attention heads.
    :returns:
        Tensor spilt by head.

        *Shape:* ``(batch_size, head, seq_len, width_per_head)``
    """
    batch_size, seq_len, model_dim = input.size()
    assert model_dim % num_heads == 0
    dims_per_head = model_dim // num_heads

    return input.view(batch_size, seq_len, num_heads, dims_per_head).transpose(1, 2)


def split_heads_grouped_qkv(
    projection: Tensor, attention_heads: AttentionHeads, dims_per_head: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Split attention heads by grouping the heads in blocks of
    the number of key/value-heads. Then distribute the heads
    across query, key, and value.

    :param projection:
        The fused query, key, value projection.
    :param attention_heads:
        Attention head configuration.
    :param dims_per_head:
        Head dimensionality.
    :returns:
        Query, key, value tensors.
    """
    num_query_heads = attention_heads._num_query_heads
    num_key_value_heads = attention_heads._num_key_value_heads

    batch_size, seq_len, _ = projection.shape

    # Group the heads in num_key_value_head-sized blocks and
    # then split the blocks.
    grouped_projection = projection.view(
        batch_size,
        seq_len,
        -1,
        num_query_heads // num_key_value_heads + 2,
        dims_per_head,
    )
    query, key, value = grouped_projection.split(
        [num_query_heads // num_key_value_heads, 1, 1], dim=-2
    )

    def permute_merge_groups(x, num_heads):
        return x.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_heads, seq_len, dims_per_head
        )

    query = permute_merge_groups(query, num_query_heads)
    key = permute_merge_groups(key, num_key_value_heads)
    value = permute_merge_groups(value, num_key_value_heads)

    return query, key, value


def combine_heads(input: Tensor) -> Tensor:
    """
    Combine the split attention head representations.

    :param input:
        Tensor split by head.

        *Shape:* ``(batch_size, head, seq_len, width_per_head)``
    :returns:
        Merged tensor.

        *Shape:* ``(batch_size, seq_len, hidden_width)``
    """
    batch_size, head, seq_len, model_dim = input.size()
    return (
        input.transpose(1, 2).contiguous().view(batch_size, seq_len, head * model_dim)
    )
