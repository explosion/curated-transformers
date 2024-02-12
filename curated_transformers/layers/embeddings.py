import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from .cache import KeyValueCache


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinusoidalPositionalEmbedding(Module):
    """
    Sinusoidal positional embeddings (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        *,
        width: int,
        max_len: int,
        normalize=True,
        device: Optional[torch.device] = None,
    ):
        """
        Construct a sinusoidal positional embedding module.

        :param width:
            Width of the embedding.
        :param max_len:
            Maximum length of the embedding.
        :param normalize:
            Perform L2 normalization of the embedding.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, width, 2, device=device) * (-math.log(10000.0) / width)
        )

        pe = torch.zeros(max_len, width, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if normalize == True:
            l2 = torch.linalg.vector_norm(pe, dim=-1)
            pe /= l2.unsqueeze(-1)

        self.pe = pe
        self.pe.requires_grad = False

    def forward(self, input: Tensor) -> Tensor:
        """
        Returns the positional embedding for the input.

        :param input:
            Input tensor.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Positional embedding for the input.

            *Shape:* ``(seq_len, width)``
        """
        return self.pe[: input.size(1), :]


class RotaryEmbeddings(Module):
    """
    Rotary embeddings (`Su et al., 2021`_).

    .. _Su et al., 2021: https://arxiv.org/abs/2104.09864
    """

    cos: Tensor
    sin: Tensor
    theta: Tensor

    def __init__(
        self,
        width: int,
        *,
        seq_len: int = 512,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ):
        """
        Construct a rotary embedding module. The rotary embedding
        will be precomputed for up to ``seq_len`` positions. The embedding
        will be recomputed when a longer sequence is found in the input.

        :param width:
            Rotary embedding width.
            Must be even.
        :param seq_len:
            Number of positions to initially precompute.
        :param base:
            The base used for :math:`\\theta_i`.
            Determines the cycle length of the embeddings.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()

        if width % 2:
            raise ValueError(f"Width of rotary embeddings must be even, was: {width}")

        # Ignore allocations on the meta device as we don't persist our buffer,
        # i.e., we don't expect the backing tensor to be replaced with pretrained weights.
        if device is not None and device.type == "meta":
            device = None
        # Θ_i = 10000^(-2(i-1)/d)
        theta = torch.pow(
            base, -torch.arange(0, width, 2, dtype=torch.float, device=device) / width
        )
        self.register_buffer("theta", theta, persistent=False)

        self._create_rotary_embed(width=width, length=seq_len)

    def _create_rotary_embed(self, *, width: int, length: int):
        # mΘ
        position = torch.arange(length, device=self.theta.device).unsqueeze(1)
        m_theta = position * self.theta.unsqueeze(0)

        # We apply both sin and cos twice (see Eq 15, 34), but the ordering
        # is changed for compatibility with most common implementations.
        m_theta = torch.cat([m_theta, m_theta], dim=-1)

        re_cos = m_theta.cos().view([length, width])
        re_sin = m_theta.sin().view([length, width])

        self.register_buffer("cos", re_cos, persistent=False)
        self.register_buffer("sin", re_sin, persistent=False)

    def _rotate(self, input: Tensor):
        """
        Rotate the input tensor by half of its innermost width.

        :param input:
            Tensor to rotate.

            *Shape:* ``(..., width)``
        :returns:
            Rotated tensor.

            *Shape:* ``(.., width)``

        :meta private:
        """
        half_idx = input.shape[-1] // 2
        input_1 = -input[..., half_idx:]
        input_2 = input[..., :half_idx]
        return torch.cat([input_1, input_2], dim=-1)

    def forward(self, input: torch.Tensor, *, positions: Optional[Tensor] = None):
        """
        Apply rotary embeddings to the input.

        :param input:
            Input to apply the rotary embeddings to.

            *Shape:* ``(batch_size, n_heads, seq_len, width_per_head)``
        :param positions:
            Positions of the inputs. If no positions are
            provided, they are assumed to be ``[0, seq_len)``.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Input with the rotary embeddings applied.

            *Shape:* ``(batch_size, n_heads, seq_len, width_per_head)``
        """
        batch_size, _, seq_len, width = input.shape

        if positions is None:
            # Fastpath: positions from [0..seq_len), avoid indexing.
            if self.cos.size(-2) < seq_len:
                self._create_rotary_embed(width=width, length=seq_len)
            rot_cos = self.cos[:seq_len, :].view(1, 1, seq_len, width)
            rot_sin = self.sin[:seq_len, :].view(1, 1, seq_len, width)
        else:
            max_len = int(positions.max()) + 1
            if self.cos.size(-2) < max_len:
                self._create_rotary_embed(width=width, length=max_len)

            # Flatten positions to index cos/sin arrays, then unflatten.
            #
            # Example shapes:
            #
            #   positions_flat - (batch_size * seq_len)
            #   self.cos - (max_len, width)
            #   rot_cos - (batch_size, seq_len, width)
            positions_flat = positions.view(-1)
            rot_cos = self.cos[positions_flat].view(batch_size, 1, seq_len, width)
            rot_sin = self.sin[positions_flat].view(batch_size, 1, seq_len, width)

        # Eq 34 with ordering changed for compatibility.
        return rot_cos * input + rot_sin * self._rotate(input)


class QueryKeyRotaryEmbeddings(Module):
    """
    Rotary embeddings (`Su et al., 2021`_) applied to query
    and key representations.

    .. _Su et al., 2021: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        *,
        base: int = 10000,
        fraction: float,
        head_width: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Construct a rotary embedding module.

        :param base:
            Base in signifying the rotary embedding period.
        :param fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        :param head_width:
            Width of key and value heads.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"Rotary embedding fraction should be between 0.0 and 1.0 inclusive, was: {fraction}"
            )
        self.rotary_width = int(fraction * head_width)
        self.head_width = head_width
        self.rotary_embeds = RotaryEmbeddings(
            width=self.rotary_width, base=base, device=device
        )

    def forward(
        self,
        *,
        query: Tensor,
        key: Tensor,
        cache: Optional[KeyValueCache] = None,
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to the query and key.

        :param query:
            Query representations.

            *Shape:* ``(batch_size, head, seq_len, width_per_head)``
        :param key:
            Key representations.

            *Shape:* ``(batch_size, head, seq_len, width_per_head)``
        :param cache: Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions: Input positions. Positions are needed to
            look up rotary embeddings. Normally, these positions are calculated
            automatically. But if the positions deviate for some reason, they
            can be provided through this argument.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Query and key with the rotary embeddings applied.

            *Shape:* ``(batch_size, head, seq_len, width_per_head)``
        """
        head_width = self.head_width
        rotary_width = self.rotary_width

        # If a cache was provided, but no positions, assume that the
        # positions of the current batch continue from the cache.
        if cache is not None and positions is None:
            cache_len = cache.key.size(-2)
            seq_len = key.size(-2)
            positions = torch.arange(
                cache_len,
                cache_len + seq_len,
                dtype=torch.long,  # `torch.int32` isn't supported in indexing operations prior in torch<2.0.0.
                device=key.device,
            ).repeat(key.size(0), 1)

        if rotary_width == head_width:
            # Fast path: we apply rotary embeddings the full key/query vectors.
            key = self.rotary_embeds(key, positions=positions)
            query = self.rotary_embeds(query, positions=positions)
        else:
            # Otherwise, split up key/query vectors, apply rotary embeddings
            # and concatenate again.
            k_rotary, k_rest = key.split([rotary_width, head_width - rotary_width], -1)
            q_rotary, q_rest = query.split(
                [rotary_width, head_width - rotary_width], -1
            )

            # Apply rotary embeddings.
            k_rotary = self.rotary_embeds(k_rotary, positions=positions)
            q_rotary = self.rotary_embeds(q_rotary, positions=positions)

            query = torch.cat([q_rotary, q_rest], dim=-1)
            key = torch.cat([k_rotary, k_rest], dim=-1)

        return query, key
