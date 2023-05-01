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

    def __init__(self, width: int, seq_len: int = 512, *, base: int = 10000):
        """Cached rotary embeddings (Su et al., 2021). The rotary embedding
        will be computed for up to 'initial_len' positions. However, the
        embedding will be recomputed when a longer sequence is found in the
        input.

        width (int):
            Rotary embedding dimensionality.
        seq_len (int):
            Number of positons to initially pre-cache.
        base (int):
            The base used for Θ_i, determines the cycle length of the
            embeddings."""
        super().__init__()

        # Θ_i = 10000^(-2(i-1)/d)
        theta = torch.pow(base, -torch.arange(0, width, 2, dtype=torch.float) / width)
        self.register_buffer("theta", theta, persistent=False)

        self._create_rotary_embed(width=width, length=seq_len)

    def _create_rotary_embed(self, *, width: int, length: int):
        # mΘ
        position = torch.arange(length).unsqueeze(1)
        m_theta = position * self.theta.unsqueeze(0)

        # We apply both sin and cos twice (see Eq 15, 34).
        m_theta = torch.cat([m_theta, m_theta], dim=-1)

        re_cos = m_theta.cos().view([1, 1, length, width])
        re_sin = m_theta.sin().view([1, 1, length, width])

        self.register_buffer("cos", re_cos, persistent=False)
        self.register_buffer("sin", re_sin, persistent=False)

    def forward(self, input: torch.Tensor):
        """
        input (Tensor): Input to return the rotary embeddings for.
        RETURNS (Tuple[Tensor, Tensor])] cosine and sine rotary embeddings.
        Shapes:
            input - (batch, head, seq_len, width_per_head)
            output - pair of (1, 1, seq_len, width_per_head)
        """
        seq_len = input.size(-2)
        if self.cos.size(-2) < seq_len:
            self._create_rotary_embed(width=self.cos.size(-1), length=seq_len)
        return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]
