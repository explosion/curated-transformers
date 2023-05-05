import math
import torch
from torch import Tensor
from torch.nn import Module


class GeluNew(Module):
    """GELU approximation, called `gelu_new` in many transformer models."""

    def forward(self, input: Tensor) -> Tensor:
        """
        Shapes:
            input - (batch, seq_len, width)
        """
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )
