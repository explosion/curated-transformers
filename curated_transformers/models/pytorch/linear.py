import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        # Work around issue with linear with the MPS backend. See:
        # https://github.com/pytorch/pytorch/issues/97239
        if hasattr(input, "is_mps") and input.is_mps:
            return torch.matmul(input, self.weight.t()) + self.bias
        else:
            return F.linear(input, self.weight, self.bias)
