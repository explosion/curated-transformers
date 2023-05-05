import torch

from .conftest import TORCH_DEVICES

TORCH_DEVICES.append(torch.device("cuda"))
