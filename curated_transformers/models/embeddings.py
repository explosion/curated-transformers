import math
import torch
from torch import Tensor
from torch.nn import Module


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinusoidalPositionalEmbedding(Module):
    def __init__(self, dim: int, max_len: int, *, normalize: bool = True):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if normalize:
            l2 = torch.linalg.vector_norm(pe, dim=-1)
            pe /= l2.unsqueeze(-1)

        pe.requires_grad = False
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, input: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len)
        """
        return self.pe[: input.size(1), :]
