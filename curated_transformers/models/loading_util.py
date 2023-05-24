import itertools
from typing import Mapping, Optional
import torch
from torch.nn import Module, Parameter


def _emplace_module_tensor(
    module: Module,
    tensor_name: str,
    tensor: torch.Tensor,
    device: Optional[torch.device] = None,
):
    """Replaces a module's parameter or (persistent) buffer with the passed tensor and moves it
    to the given device. This is a zero-copy operation (excluding D2H/H2D transfers) where the
    input tensor is directly associated with the module. Unexpected behaviour can occur if the same
    tensor is associated with multiple modules.

    :param module: Module whose associated tensor needs replacing.
    :param tensor_name: Name of the tensor to be replaced.
    :param tensor: eplacement tensor.
    :param device: Device to which the replacement tensor is moved.
    """
    is_parameter = tensor_name in module._parameters
    is_buffer = tensor_name in module._buffers
    assert is_parameter ^ is_buffer

    if device is not None:
        tensor = tensor.to(device=device)

    size_mismatch_msg = (
        "Expected the size of the replacement for `{}` to be {}, but got {}"
    )

    if is_parameter:
        old_param = module._parameters[tensor_name]
        assert old_param is not None
        if old_param.shape != tensor.shape:
            raise ValueError(
                size_mismatch_msg.format(tensor_name, old_param.shape, tensor.shape)
            )
        new_param = Parameter(tensor, requires_grad=old_param.requires_grad)
        module._parameters[tensor_name] = new_param
    else:
        old_buffer = module._buffers[tensor_name]
        assert old_buffer is not None
        if old_buffer.shape != tensor.shape:
            raise ValueError(
                size_mismatch_msg.format(tensor_name, old_buffer.shape, tensor.shape)
            )
        module._buffers[tensor_name] = tensor


def emplace_module_state_dict(
    module: Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    device: Optional[torch.device] = None,
):
    """Recursively replace a module's parameters and buffers with the tensors in the given
    `state_dict` without performing any unnecessary copies.

    :param module: Top-level module whose child tensors needs replacing.
    :param state_dict: State dictionary containing the replacement tensors
        and their names.
    :param device: Device to which the replacement tensors are moved.
    """

    def emplace(module: Module, state_dict: Mapping[str, torch.Tensor], prefix: str):
        persistent_buffers = {
            k: v
            for k, v in module._buffers.items()
            if k not in module._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            module._parameters.items(), persistent_buffers.items()
        )

        for name, param in local_name_params:
            key = prefix + name
            if param is None:
                if key in state_dict:
                    raise ValueError(
                        f"Key `{name}` found in state dict but no data in module `{prefix}`"
                    )
                else:
                    continue
            if key not in state_dict:
                raise ValueError(f"Key `{name}` not found in module `{prefix}`")
            replacement = state_dict[key]
            _emplace_module_tensor(module, name, replacement, device=device)

    def traverse(module: Module, state_dict: Mapping[str, torch.Tensor], prefix: str):
        emplace(module, state_dict, prefix)

        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {
                    k: v for k, v in state_dict.items() if k.startswith(child_prefix)
                }
                traverse(child, child_state_dict, child_prefix)

    traverse(module, state_dict, "")
    del traverse
    del emplace
