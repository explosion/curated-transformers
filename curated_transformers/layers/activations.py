import math
from enum import Enum
from typing import Type

import torch
from torch import Tensor
from torch.nn import Module


class Activation(Enum):
    """
    Activation functions.

    .. _Black et al., 2022: https://arxiv.org/abs/2204.06745
    .. _Hendrycks et al., 2016: https://arxiv.org/abs/1606.08415
    .. _Fukushima, 1969: https://ieeexplore.ieee.org/document/4082265
    """

    #: Rectified Linear Unit (`Fukushima, 1969`_).
    ReLU = "relu"

    #: Gaussian Error Linear Unit (`Hendrycks et al., 2016`_).
    GELU = "gelu"

    #: Gaussian Error Linear Unit (`Hendrycks et al., 2016`_) approximation
    #: used by GPT-NeoX (`Black et al., 2022`_).
    GELUFast = "gelu_fast"

    #: Gaussian Error Linear Unit (`Hendrycks et al., 2016`_) approximation.
    GELUNew = "gelu_new"

    #: Sigmoid Linear Unit (`Hendrycks et al., 2016`_).
    SiLU = "silu"

    @classmethod
    def _missing_(cls, value):
        supported_activations = ", ".join(sorted(v.value for v in cls))
        raise ValueError(
            f"Invalid activation function `{value}`. "
            f"Supported functions: {supported_activations}"
        )

    @property
    def module(self) -> Type[torch.nn.Module]:
        """
        Get the PyTorch module for the activation function.
        """

        # TODO: convert to match when Python 3.10 is the minimum version.
        if self == Activation.ReLU:
            return torch.nn.ReLU
        elif self == Activation.GELU:
            return torch.nn.GELU
        elif self == Activation.GELUFast:
            return GELUFast
        elif self == Activation.GELUNew:
            return GELUNew
        elif self == Activation.SiLU:
            return torch.nn.SiLU
        else:
            raise NotImplementedError


class GELUNew(Module):
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


class GELUFast(Module):
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
        alpha = math.sqrt(2.0 / math.pi)
        beta = 0.044715

        return 0.5 * input * (1.0 + torch.tanh(alpha * (input + beta * input.pow(3))))
