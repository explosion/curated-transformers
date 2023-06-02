import torch

from . import conftest

conftest.TORCH_DEVICES.append(torch.device("cuda", index=0))
conftest.GPU_TESTS_ENABLED = True
