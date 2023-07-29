import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module

from ..util.dataclass import DataclassAsDict
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


@dataclass
class AttentionMask(DataclassAsDict):
    """
    Mask for attention calculation. Sequence elements for which the
    corresponding mask element is set to ``False`` are ignored during
    attention calculation.

    :param bool_mask:
        The boolean mask.
    """

    bool_mask: Tensor

    def __init__(self, bool_mask: Tensor):
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

        self.__post_init__()

    @classmethod
    def jit_rewrap(
        cls: Type["AttentionMask"],
        attention_mask: Optional[Union["AttentionMask", Dict[str, Tensor]]],
    ) -> Optional["AttentionMask"]:
        """
        Rewrap TorchScript dictionary conversion of an attention mask
        as an ``AttentionMask``.

        :param attention_mask:
            The attention mask or its dictionary representation. If the
            value is an ``AttentionMask`` or ``None``, the value will be
            returned as-is.
        :returns:
            The attention mask.
        """
        if attention_mask is None or isinstance(attention_mask, AttentionMask):
            return attention_mask

        bool_mask = attention_mask.get("bool_mask")
        if bool_mask is None:
            raise ValueError(
                "Attention mask is not of the `AttentionMask` type, nor a dict with 'bool_mask'."
            )
        return AttentionMask(bool_mask=bool_mask)

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

    def filter_batch_items(self, mask: Tensor) -> "AttentionMask":
        """
        Filter batch sequences from the attention mask.

        Sequences for which the mask is ``True`` are retained.

        :param mask:
            Mask of batch items to retain.

            *Shape:* ``(batch_size,)``
        :returns:
            Filtered mask.
        """
        if mask.ndim != 1:
            raise ValueError(
                f"Attention mask filter must be a 1D tensor, has {mask.ndim} dimensions."
            )
        if mask.size(0) != self.bool_mask.size(0):
            raise ValueError(
                f"Attention mask filter size ({mask.size(0)}) must match attention mask batch size ({self.bool_mask.size(0)})."
            )
        if mask.dtype != torch.bool:
            raise ValueError(
                f"Attention mask filter dtype must be bool, was: {mask.dtype}."
            )

        return AttentionMask(self.bool_mask[mask])

    def dim(self) -> int:
        """
        Return the number of dimensions in the mask.
        """
        return self.bool_mask.dim()

    def merge_mask(self, other: "AttentionMask") -> "AttentionMask":
        """
        Merge this attention mask with another attention mask.

        :param other:
            Attention mask to merge.
        :returns:
            Merged mask.
        """
        return AttentionMask(self.bool_mask.logical_and(other.bool_mask))

    def logit_mask(self, dtype: torch.dtype) -> Tensor:
        """
        Generate the logit mask for the given ``dtype``.

        Elements of the mask that are ``False`` are set to the
        minimum value of the ``dtype`` and the rest to zero. During
        softmax calculation, adding this mask to the logits will
        result in (near-)zero probabilities for the elements that
        are ``False``.

        :param dtype:
            Data type of the logit mask.
        :returns:
            Logit mask.
        """
        return (1.0 - self.bool_mask.to(dtype)) * torch.finfo(dtype).min

    def extend_length(self, count: int, fill_value: bool) -> "AttentionMask":
        """
        Extend the attention mask in the sequence length dimension
        by the given value.

        :param count:
            Number of new elements to insert.
        :param fill_value:
            Value to store in the new elements.
        :returns:
            Extended mask.
        """
        if count <= 0:
            raise ValueError(
                f"Attention mask sequence length extension requires a non-zero, positive value for 'count'"
            )

        ext = torch.full(
            self.shape[:3] + (count,),
            fill_value,
            device=self.device,
        )
        return AttentionMask(torch.cat([self.bool_mask, ext], dim=-1))

    @property
    def shape(self) -> torch.Size:
        """
        Return the shape of the mask.
        """
        return self.bool_mask.shape

    @property
    def device(self) -> torch.device:
        """
        Return the device of the mask.
        """
        return self.bool_mask.device


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
        torch.ones((key_len, key_len), device=query.device, dtype=torch.bool),
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
        Construct a multi-query attention configuration: key has one head,
        value has one head, query has ``num_query_heads`` heads
        (`Shazeer et al., 2019`_). The key head and the value head are
        broadcast to the shape of the query.

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


class QkvMode(Enum):
    """
    How the query, key and value projections are handled in
    the self-attention layer.
    """

    #: ``SEPARATE`` - Use separate projections for query, key and value.
    SEPARATE = 0

    #: ``MERGED_SPLIT_BEFORE`` - Use a merged projection for query,
    #: key and value, and split heads before splitting the query, key
    #: and value representations. This ordering is incompatible with
    #: head sharing in keys or values.

    MERGED_SPLIT_BEFORE = 1

    #: ``MERGED_SPLIT_AFTER`` - Use a merged projection for query,
    #: key and value, and split heads after splitting the query, key
    #: and value representations.
    MERGED_SPLIT_AFTER = 2


class AttentionLinearBiases(Module):
    """
    ALiBi: Linear biases for attention (`Press et al., 2022`_).

    .. _Press et al., 2022: https://arxiv.org/abs/2108.12409
    """

    slopes: Tensor

    def __init__(
        self, *, num_attention_heads: int, is_causal: bool, is_inverted: bool
    ) -> None:
        """
        Construct an ALiBi module.

        :param num_attention_heads:
            Number of attention heads.
        :param is_causal:
            Use causal attention.
        :param invert:
            If ``True``, the biases are inverted, i.e.,
            penalties become rewards.
        """
        super().__init__()

        self.is_causal = is_causal
        self.invert = is_inverted
        slopes = self._calculate_slopes(num_attention_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    def _calculate_slopes(self, num_attention_heads: int) -> Tensor:
        """
        Calculate the linear bias slopes for a given number
        of attention heads.

        :param num_attention_heads:
            Number of attention heads.
        :returns:
            Head slope tensor.

            *Shape:* ``(1, heads, 1, 1)``

        :meta private:
        """

        def _slopes_with_step(num_attention_heads, *, step=1):
            ratio = 2.0 ** (-8.0 / num_attention_heads)
            return ratio ** torch.arange(1, 1 + num_attention_heads, step)

        # The slope as proposed in the ALiBi paper would be:
        #
        # return _slopes_with_step(num_attention_heads)
        #
        # However the authors note in their implementation that using powers
        # of 2 for n in the ratio 2**(-8/n) of the geometric sequence for
        # slopes has better properties.
        #
        # Most implementations use powers of two in the following
        # manner: if the number of heads is not a power of 2, then we find
        # k=the largest power of 2 in 1..n. The slopes are then computed
        # as the concatenation of:
        #
        # - The slopes for 1..k.
        # - The slopes for 1..2*k with step 2, taking the first n-k elements.

        # k is the largest power of 2 in 1..n.
        k = 1 << ((num_attention_heads).bit_length() - 1)
        slopes = _slopes_with_step(k)

        if num_attention_heads != k:
            remaining_heads = num_attention_heads - k
            slopes_rest = _slopes_with_step(2 * k, step=2)[:remaining_heads]
            slopes = torch.cat([slopes, slopes_rest])

        return slopes.view(1, -1, 1, 1)

    def calculate_biases(self, seq_len: int) -> Tensor:
        """
        Calculate the linear bias tensor upto a given (key) sequence length.

        :param seq_len:
            Maximum number of timesteps to calculate.
        :returns:
            Multi-headed linear bias tensor.

            *Shape:* ``(1, heads, seq_len, seq_len)`` (non-causal) or
            ``(1, heads, 1, seq_len)`` (causal)

        :meta private:
        """
        if self.is_causal:
            distances = torch.arange(1 - seq_len, 1)
        else:
            distances = torch.arange(seq_len) - torch.arange(seq_len).view(-1, 1)
            distances = distances.abs().mul(-1).view(1, 1, seq_len, seq_len)

        if self.invert:
            distances += seq_len - 1

        return distances * self.slopes

    def forward(self, *, attention_scores: Tensor, inplace: bool = True) -> Tensor:
        """
        Apply linear biases to (unmasked) attention scores.

        :param attention_scores:
            Attention scores.

            *Shape:* ``(batch_size, heads, query_len, key_len)``
        :param inplace:
            Update attention scores inplace.
        :returns:
            Attention scores with linear biases.

            *Shape:* ``(batch_size, heads, query_len, key_len)``
        """
        if not inplace:
            attention_scores = attention_scores.clone()

        biases = self.calculate_biases(attention_scores.size(-1)).to(
            dtype=attention_scores.dtype, device=attention_scores.device
        )
        return attention_scores + biases


class ScaledDotProductAttention(Module):
    """
    Scaled dot-product attention (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    linear_biases: Optional[AttentionLinearBiases]

    def __init__(
        self, *, dropout_prob: float, linear_biases: Optional[AttentionLinearBiases]
    ):
        """
        Construct a scaled dot-product attention module.

        :param dropout_prob:
            Dropout to apply to the final hidden representation.
        :param linear_biases:
            ALiBi (`Press et al., 2022`_) for attention scores.
            Not applied if ``None``.

        .. _Press et al., 2022: https://arxiv.org/abs/2108.12409
        """
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.linear_biases = linear_biases

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

        :param query:
            Query tensor.

            *Shape:* ``(batch_size, heads, seq_len, width)``
        :param key:
            Key tensor.

            *Shape:* ``(batch_size, heads, seq_len, width)``
        :param value:
            Value tensor.

            *Shape:* ``(batch_size, heads, seq_len, width)``
        :param attention_mask:

            Attention mask. Sequence elements for which the corresponding mask
            element is set to ``False`` are ignored in attention.
        :returns:
            Attention values.

            *Shape:* ``(batch_size, heads, seq_len, width)``
        """
        model_dim = key.shape[-1]
        attn_scores = query @ key.transpose(-2, -1)
        attn_scores /= math.sqrt(model_dim)

        if self.linear_biases is not None:
            attn_scores = self.linear_biases(attention_scores=attn_scores)

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
        attention_biases: Optional[AttentionLinearBiases] = None,
        use_bias: bool,
        device: Optional[torch.device] = None,
    ):
        """
        Construct a self-attention layer with rotary position embeddings
        and attention linear biases.

        :param attention_heads:
            Attention head configuration.
        :param dropout_prob:
            Dropout to apply between the self-attention and output layers.
        :param hidden_width:
            Hidden width of the layer.
        :param attention_biases:
            ALiBi biases. ALiBi will not be used when set to ``None``.
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
        self.use_alibi = attention_biases is not None

        self.rotary_embeds = rotary_embeds

        self.attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob, linear_biases=attention_biases
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
        # The attention mask is converted to a dict for traced models. Rewrap as
        # AttentionMask to get validation and utility methods.
        attention_mask = AttentionMask.jit_rewrap(attention_mask)

        query, key, value = self._query_key_value(input)

        if self.rotary_embeds is not None:
            query, key = self.rotary_embeds(
                query=query, key=key, cache=cache, positions=positions
            )

        # The key-value is converted to a dict for traced models. Rewrap as
        # KeyValueCache to get validation and utility methods.
        cache = KeyValueCache.jit_rewrap(cache)
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
            attn_mask = (
                None if combined_mask is None else combined_mask.logit_mask(query.dtype)
            )

            # Add AliBi to the logit mask
            if self.use_alibi:
                assert self.attention.linear_biases is not None
                biases = self.attention.linear_biases.calculate_biases(key.size(-2)).to(
                    dtype=query.dtype, device=query.device
                )
                if combined_mask is not None:
                    bool_mask = combined_mask.bool_mask
                    assert attn_mask is not None
                    attn_mask = torch.where(bool_mask, biases, attn_mask)
                else:
                    # Just pass the ALiBi biases.
                    attn_mask = biases

            # We can't pass a bool mask, because it is currently broken:
            # https://github.com/pytorch/pytorch/issues/103749
            attn = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
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
            return output, KeyValueCache(key, value)
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

        *Shape:* ``(batch_size, seq_len, (num_query_heads + 2 * num_key_value_heads) * dims_per_head)``
    :param attention_heads:
        Attention head configuration.
    :param dims_per_head:
        Head dimensionality.
    :returns:
        Query, key, value tensors.

        *Shapes:*
        - Query: ``(batch_size, num_query_heads, seq_len, dims_per_head)``
        - Key: ``(batch_size, num_key_value_heads, seq_len, dims_per_head)``
        - Value: ``(batch_size, num_key_value_heads, seq_len, dims_per_head)``
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
