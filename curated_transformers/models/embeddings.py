import math
import torch
from torch import Tensor
from torch.nn import Module


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinusoidalPositionalEmbedding(Module):
    def __init__(self, dim: int, max_len: int, *, normalize=True):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if normalize == True:
            l2 = torch.norm(pe, dim=-1)
            pe /= l2.unsqueeze(-1)

        self.pe = pe
        self.pe.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len)
        """
        return self.pe[x.size(1), :]


class RotaryEmbeddings(Module):
    cos: Tensor
    sin: Tensor
    theta: Tensor

    def __init__(self, width: int, *, seq_len: int = 512, base: int = 10000):
        """Rotary embeddings (Su et al., 2021) layer. The rotary embedding
        will be precomputed for up to 'seq _len' positions. The embedding
        will be recomputed when a longer sequence is found in the input.

        width (int):
            Rotary embedding dimensionality, must be even.
        seq_len (int):
            Number of positons to initially precompute.
        base (int):
            The base used for Θ_i, determines the cycle length of the
            embeddings."""
        super().__init__()

        if width % 2:
            raise ValueError(f"Width of rotary embeddings must be even, was: {width}")

        # Θ_i = 10000^(-2(i-1)/d)
        theta = torch.pow(base, -torch.arange(0, width, 2, dtype=torch.float) / width)
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
        """Rotate the input tensor by half of its innermost width.

        input (Tensor): array to rotate.
        RETURNS (Tensor): rotated array.

        Shapes:
            input - (..., width)
            output - (..., width)
        """
        half_idx = input.shape[-1] // 2
        input_1 = -input[..., half_idx:]
        input_2 = input[..., :half_idx]
        return torch.cat([input_1, input_2], dim=-1)

    def forward(self, input: torch.Tensor):
        """
        Apply rotary embeddings to an array.

        input (Tensor): Array to apply the rotary embeddings to.
        RETURNS (Tensor]: Array with the rotary embeddings applied.

        Shapes:
            input - (..., seq_len, width_per_head)
            output - pair of (..., seq_len, width_per_head)
        """
        seq_len = input.size(-2)
        width = self.cos.size(-1)
        if self.cos.size(-2) < seq_len:
            self._create_rotary_embed(width=width, length=seq_len)
        shape = ((1,) * (input.ndim - self.cos.ndim)) + (seq_len, width)
        rot_cos = self.cos[:seq_len, :].view(shape)
        rot_sin = self.sin[:seq_len, :].view(shape)

        # Eq 34 with ordering changed for compatibility.
        return rot_cos * input + rot_sin * self._rotate(input)