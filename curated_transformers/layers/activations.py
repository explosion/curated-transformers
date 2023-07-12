import math

import torch
from torch import Tensor
from torch.nn import Module


class GeluNew(Module):
    """
    GELU (`Hendrycks et al., 2016`_) approximation, called ``gelu_new`` in many transformer models.

    .. _Hendrycks et al., 2016: https://arxiv.org/abs/1606.08415
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
    GELU (`Hendrycks et al., 2016`_) approximation used by GPT-NeoX (`Black et al., 2022`_).

    .. _Hendrycks et al., 2016: https://arxiv.org/abs/1606.08415
    .. _Black et al., 2022: https://arxiv.org/abs/2204.06745
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
