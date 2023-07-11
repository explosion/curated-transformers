import math

import torch
from torch import Tensor
from torch.nn import Module


class GeluNew(Module):
    """
    GELU approximation, called `gelu_new` in many transformer models.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply the GELU activation on the input.

        :param input:
            Input tensor.

            *Shape:* ``(batch_size, seq_len, width)``
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


class GeluFast(Module):
    """
    GELU approximation used by GPT-NeoX (EleutherAI/gpt-neox-20b).

    See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply the GELU activation on the input.

        :param input:
            Input tensor.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input))
            )
        )
