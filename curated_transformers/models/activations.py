import math
from typing import Callable
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


def _get_activation(name: str) -> Callable[[Tensor], Tensor]:
    if name == "relu":
        return F.relu
    elif name == "gelu":
        return F.gelu
    elif name == "gelu_new":
        # Ideally, we would use torch.nn.GELU(approximate="tanh"). However,
        # the differences between that and the manual Torch implementation
        # are large enough to fail tests comparing output to HF
        # transformers.
        return GeluNew()
    else:
        raise ValueError(f"unsupported activation function '{config.hidden_act}")


class GeluNew(Module):
    """GELU approximation, called `gelu_new` in many transformer models."""

    def forward(self, input: Tensor):
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
