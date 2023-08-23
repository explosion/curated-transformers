import functools
from contextlib import contextmanager
from typing import List

import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from .pytorch import ModuleIterator, apply_to_module


@contextmanager
def use_nvtx_ranges_for_forward_pass(module: Module):
    """
    Recursively applies `NVTX ranges`_ to the forward pass operation
    of the provided module. The ranges will be recorded during an `Nsight`_
    profiling session.

    :param module:
        Top-level module to which the ranges are applied recursively.

    .. _Nsight: https://developer.nvidia.com/nsight-systems
    .. _NVTX ranges: https://pytorch.org/docs/stable/cuda.html#nvidia-tools-extension-nvtx
    """

    hooks: List[RemovableHandle] = []

    def hook_forward(itr: ModuleIterator):
        range_name = f"{itr.name} : {type(itr.module).__name__}"

        def push(*args, _range_name: str, **kwargs):
            torch.cuda.nvtx.range_push(_range_name)

        def pop(*args, **kwargs):
            torch.cuda.nvtx.range_pop()

        forward_pre = itr.module.register_forward_pre_hook(
            functools.partial(push, _range_name=range_name)
        )
        forward_post = itr.module.register_forward_hook(pop)
        hooks.append(forward_pre)
        hooks.append(forward_post)

    try:
        apply_to_module(module, hook_forward)
        yield
    finally:
        for hook in hooks:
            hook.remove()
