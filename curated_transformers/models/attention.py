from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from torch.nn import Linear, Module

from .embeddings import RotaryEmbeddings


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


@dataclass
class KeyValueCache:
    """Cache type for layers that cache keys and values."""

    key: Tensor
    value: Tensor


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
            # The value is `torch.finfo(attn_scores.dype).min`. Unfortunately,
            # we cannot use `torch.finfo` in TorchScript.
            blocked_value = -3.4028234663852886e38
            seq_len = attn_scores.shape[-1]
            # TODO: We may want to cache this, but we should probably find out
            # if it is worth it first.
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), blocked_value, device=attn_scores.device
                ),
                diagonal=1,
            ).view(1, 1, seq_len, seq_len)
            k_len = k.shape[-2]
            q_len = q.shape[-2]
            causal_mask = causal_mask[:, :, k_len - q_len : k_len, :k_len]
            attn_scores += causal_mask

        if attention_mask is not None:
            # Replace tokens that we don't want to attend to with a large
            # negative value to zero them out during softmax normalization.
            batch, seq_len = attention_mask.shape
            attn_scores += attention_mask.logit_mask.view(batch, 1, 1, seq_len)

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

        :param dropout_prob: Dropout to apply between the self-attention
            and output layers.
        :param hidden_width: Hidden width of the layer.
        :param num_attention_heads: Number of attention heads.
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
        self.attention = ScaledDotProductAttention(dropout_prob=dropout_prob)
        self.input = Linear(self.model_dim, self.model_dim * 3)
        self.output = Linear(self.model_dim, self.model_dim)

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

        # Project query, key, and value all at once and then split. We do
        # the projection this way to have the option of switching to fused
        # PyTorch kernels in the future.
        proj = self.input(x)
        q, k, v = proj.chunk(3, dim=-1)

        # (batch, head, seq_len, width_per_head)
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
        dropout_prob: float = 0.1,
        hidden_width: int = 768,
        num_attention_heads: int = 12,
        rotary_fraction: float = 1.0,
        rotary_base: int = 10000,
        split_heads_before_chunk: bool = False,
    ):
        """Construct a self-attention layer with rotary position embeddings.

        :param dropout_prob: Dropout to apply between the self-attention
            and output layers.
        :param hidden_width: Hidden width of the layer.
        :param num_attention_heads: Number of attention heads.
        :param rotary_fraction: fraction of hidden width to apply rotary
            embeddings to. Must be in [0,1].
        :param rotary_base: Base in signifying the rotary embedding period.
        :param split_heads_before_chunk: Split the hidden representation
            into heads before splitting query/key/value representations when
            ``True``. This option is required for some newer transformer
            architectures that split heads before chunking.
        """

        super().__init__()

        self.model_dim = hidden_width
        self.num_heads = num_attention_heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"The hidden width of the transformer ({self.model_dim}) must be "
                f"divisible by the number of self-attention heads ({self.num_heads})"
            )

        self.split_heads_before_chunk = split_heads_before_chunk
        self.dims_per_head = self.model_dim // self.num_heads

        if not 0.0 <= rotary_fraction <= 1.0:
            raise ValueError(
                f"Rotary embedding fraction should be between 0.0 and 1.0 inclusive, was: {rotary_fraction}"
            )
        self.rotary_dims = int(rotary_fraction * self.dims_per_head)

        self.input = Linear(self.model_dim, self.model_dim * 3)
        self.attention = ScaledDotProductAttention(dropout_prob=dropout_prob)
        self.output = Linear(self.model_dim, self.model_dim)
        self.rotary_embeds = RotaryEmbeddings(width=self.rotary_dims, base=rotary_base)

    def forward(
        self,
        x: Tensor,
        attention_mask: AttentionMask,
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

        proj = self.input(x)

        if self.split_heads_before_chunk:
            proj = split_heads(proj, self.num_heads)
            q, k, v = proj.chunk(3, dim=-1)
        else:
            q, k, v = proj.chunk(3, dim=-1)

            # (batch, head, seq_len, width_per_head)
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
                    dtype=torch.int32,
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
