import itertools
from typing import Dict, Mapping, Optional, Union
import torch
from torch.nn import Module, Parameter


def _emplace_module_tensor(
    module: Module,
    tensor_name: str,
    prefix: str,
    tensor: torch.Tensor,
    device: Optional[torch.device] = None,
):
    """Replaces a module's parameter or (persistent) buffer with the passed tensor and moves it
    to the given device. This is a zero-copy operation (excluding D2H/H2D transfers) where the
    input tensor is directly associated with the module. Unexpected behaviour can occur if the same
    tensor is associated with multiple modules.

    :param module: Module whose associated tensor needs replacing.
    :param tensor_name: Name of the tensor to be replaced.
    :param prefix: Module prefix.
    :param tensor: eplacement tensor.
    :param device: Device to which the replacement tensor is moved.
    """
    is_parameter = tensor_name in module._parameters
    is_buffer = tensor_name in module._buffers
    assert is_parameter ^ is_buffer

    if device is not None:
        tensor = tensor.to(device=device)

    full_name = f"{prefix}{tensor_name}"
    if is_parameter:
        old_param = module._parameters[tensor_name]
        assert old_param is not None
        _validate_replacement(old_param, tensor, full_name)
        new_param = Parameter(tensor, requires_grad=old_param.requires_grad)
        module._parameters[tensor_name] = new_param
    else:
        old_buffer = module._buffers[tensor_name]
        assert old_buffer is not None
        _validate_replacement(old_buffer, tensor, full_name)
        module._buffers[tensor_name] = tensor


def _validate_replacement(
    replaced: Union[Parameter, torch.Tensor],
    replacement: torch.Tensor,
    name: str,
):
    if replaced.shape != replacement.shape:
        raise ValueError(
            f"Expected size of replacement for `{name}` to be {replaced.shape}, but got {replacement.shape}"
        )
    elif replaced.dtype != replacement.dtype:
        raise ValueError(
            f"Expected dtype of replacement for `{name}` to be {replaced.dtype}, but got {replacement.dtype}"
        )


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

    queue = [(module, state_dict, "")]
    while queue:
        current_module, current_state_dict, prefix = queue.pop(0)

        local_params: Dict[
            str, Union[Optional[Parameter], Optional[torch.Tensor]]
        ] = dict(current_module._parameters.items())
        for name, buf in current_module._buffers.items():
            if name in local_params:
                raise KeyError(
                    f"Key `{name}` used in both learnable parameters and buffers in module `{prefix}`"
                )
            elif name not in current_module._non_persistent_buffers_set:
                local_params[name] = buf

        for name, param in local_params.items():
            key = prefix + name
            if param is None:
                if key in current_state_dict:
                    raise ValueError(
                        f"Key `{name}` found in state dict but no data in module `{prefix}`"
                    )
                else:
                    continue
            if key not in current_state_dict:
                raise ValueError(f"Key `{name}` not found in module `{prefix}`")
            replacement = current_state_dict[key]
            _emplace_module_tensor(
                current_module, name, prefix, replacement, device=device
            )

        for name, child in current_module._modules.items():
            if child is not None:
                child_prefix = f"{prefix}{name}."
                child_state_dict = {
                    k: v for k, v in state_dict.items() if k.startswith(prefix)
                }
                queue.append((child, child_state_dict, child_prefix))
