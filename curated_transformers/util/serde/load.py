from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Mapping, Optional, Set, Union

import torch
from torch.nn import Module, Parameter

from ...repository.file import RepositoryFile
from ..pytorch import ModuleIterator, apply_to_module
from .checkpoint import ModelCheckpointType

# Args: Parent module, module prefix, parameter name, tensor to convert, device.
# Returns the new paramater.
TensorToParameterConverterT = Callable[
    [Module, str, str, torch.Tensor, Optional[torch.device]], Parameter
]

# Args: State dict.
# Returns the converted state dict.
HFStateDictConverterT = Callable[
    [Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]
]


def load_model_from_checkpoints(
    model: Module,
    *,
    filepaths: Iterable[RepositoryFile],
    checkpoint_type: ModelCheckpointType,
    state_dict_converter: HFStateDictConverterT,
    tensor_to_param_converter: Optional[TensorToParameterConverterT] = None,
    device: Optional[torch.device] = None,
):
    """
    Load parameters from PyTorch checkpoints with minimal copies.

    :param model:
        PyTorch module into which the parameters are to be loaded.
    :param filepaths:
        Paths to PyTorch checkpoints.
    :param checkpoint_type:
        Type of checkpoint being loaded.
    :param state_dict_converter:
        Callback to convert Hugging Face state dicts to the
        ``curated-transformers`` format.
    :param tensor_to_param_converter:
        Callback to perform custom conversions of the loaded parameters.
        Useful for loading quantized weights.
    :param device:
        Device in which to place the loaded parameters.
    """

    state_dicts = checkpoint_type.loader(filepaths)
    # We need to cache the model's parameter keys before loading the state
    # dicts as the process could potentially change the structure of sub-modules,
    # e.g: when quantized layers rename their parameters.
    module_keys = set(model.state_dict().keys())
    seen_keys: Set[str] = set()

    for state_dict in state_dicts:
        converted = state_dict_converter(state_dict)
        if len(converted) == 0:
            continue
        seen_keys.update(converted.keys())

        # We have to walk the module tree for each state dict as there
        # are no guarantees on the ordering of the keys.
        _emplace_module_state_dict(
            model,
            converted,
            tensor_to_param_converter=tensor_to_param_converter,
            device=device,
        )

    # Make sure that we didn't miss any keys.
    missing_keys = module_keys.difference(seen_keys)
    if len(missing_keys) != 0:
        raise ValueError(f"Some parameters were not updated/replaced: {missing_keys}")


def default_tensor_to_parameter_converter(
    module: Module,
    module_prefix: str,
    parameter_name: str,
    tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Parameter:
    """
    Default tensor to parameter converter.

    :param module:
        Parent module of the parameter being converted/replaced.
    :param module_prefix:
        Prefix of the parent module.
    :param parameter_name:
        Name of the parameter being converted/replaced.
    :param tensor:
        Tensor to be converted.
    :param device:
        Device to which the converted parameter is moved.
    :returns:
        Converted parameter.
    """
    old_param = module._parameters[parameter_name]
    assert old_param is not None
    _validate_replacement(old_param, tensor, module_prefix)
    return Parameter(tensor.to(device=device), requires_grad=old_param.requires_grad)  # type: ignore


def _emplace_module_state_dict(
    module: Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    tensor_to_param_converter: Optional[TensorToParameterConverterT] = None,
    device: Optional[torch.device] = None,
):
    if tensor_to_param_converter is None:
        tensor_to_param_converter = default_tensor_to_parameter_converter

    def apply(itr: ModuleIterator):
        prefix_with_dot = f"{itr.prefix}."
        candidate_tensors = {
            k: v for k, v in state_dict.items() if k.startswith(prefix_with_dot)
        }
        if len(candidate_tensors) == 0:
            return

        local_params_and_buffers: Dict[
            str, Union[Optional[Parameter], Optional[torch.Tensor]]
        ] = dict(itr.module._parameters.items())
        for name, buf in itr.module._buffers.items():
            if name in local_params_and_buffers:
                raise KeyError(
                    f"Key `{name}` used in both learnable parameters and buffers in module `{itr.prefix}`"
                )
            elif name not in itr.module._non_persistent_buffers_set:
                local_params_and_buffers[name] = buf

        for name, param in local_params_and_buffers.items():
            key = f"{prefix_with_dot}{name}"
            if key not in candidate_tensors:
                continue
            elif param is None:
                raise ValueError(
                    f"Key `{name}` found in state dict but no data in module `{itr.prefix}`"
                )
            replacement = candidate_tensors[key]
            assert tensor_to_param_converter is not None
            _emplace_module_tensor(
                module=itr.module,
                module_prefix=itr.prefix,
                tensor_name=name,
                replacement_tensor=replacement,
                tensor_to_param_converter=tensor_to_param_converter,
                device=device,
            )

    apply_to_module(module, apply)


def _emplace_module_tensor(
    module: Module,
    module_prefix: str,
    tensor_name: str,
    replacement_tensor: torch.Tensor,
    tensor_to_param_converter: TensorToParameterConverterT,
    device: Optional[torch.device] = None,
):
    """
    Replace a module's parameter or (persistent) buffer with the
    passed tensor and moves it to the given device.

    This is a zero-copy operation (excluding D2H/H2D transfers) where
    the input tensor is directly associated with the module. Unexpected
    behaviour can occur if the same tensor is associated with multiple modules.
    """
    is_parameter = tensor_name in module._parameters
    is_buffer = tensor_name in module._buffers
    assert is_parameter ^ is_buffer

    if is_parameter:
        new_param = tensor_to_param_converter(
            module, module_prefix, tensor_name, replacement_tensor, device
        )
        module._parameters[tensor_name] = new_param
    else:
        old_buffer = module._buffers[tensor_name]
        assert old_buffer is not None
        _validate_replacement(
            old_buffer, replacement_tensor, f"{module_prefix}.{tensor_name}"
        )
        module._buffers[tensor_name] = replacement_tensor


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
