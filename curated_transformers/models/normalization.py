from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class RMSNorm(Module):
    """
    Root Mean Square Normalization (Zhang & Sennrich, 2019).
    """

    def __init__(self, dim: int, *, eps: float, device: Optional[torch.device] = None):
        """
        :param dim:
            The (hidden) dimensionality of the representations that RMS
            normalization will be applied to.
        :param eps:
            Epsilon to avoid division by zero.
        :param device: Device on which the module is to be initialized.
        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones((dim,), device=device))

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply RMS normalization to a tensor.

        :param input:
            The tensor to apply normalization to.
        :returns:
            Normalized tensor.
        """
        # Zhang & Sennrich, Equation 4
        rms = (
            input.to(torch.float32)
            .square()
            .mean(-1, keepdim=True)
            .add(self.eps)
            .rsqrt()
        )

        return (input * rms).to(input.dtype) * self.weight
