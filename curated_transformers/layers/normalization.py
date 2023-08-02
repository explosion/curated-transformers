from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class RMSNorm(Module):
    """
    Root Mean Square (RMS) normalization (`Zhang et al., 2019`_).

    .. _Zhang et al., 2019: https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self, width: int, *, eps: float, device: Optional[torch.device] = None
    ):
        """
        Construct a RMS normalization module.

        :param width:
            The (hidden) width of the representations that RMS
            normalization will be applied to.
        :param eps:
            Epsilon to avoid division by zero.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones((width,), device=device))

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply RMS normalization to a tensor.

        :param input:
            The tensor to apply normalization to.
        :returns:
            Normalized tensor.
        """
        # Zhang & Sennrich, Equation 4. If we are in lower precision than
        # float32, then squaring and averaging can get way off. So for
        # normalization we want to use higher precision.
        rms = (
            input.to(torch.float32)
            .square()
            .mean(-1, keepdim=True)
            .add(self.eps)
            .rsqrt()
        )

        return (input * rms).to(input.dtype) * self.weight
