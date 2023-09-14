from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Set,
    Union,
)

import torch
from fsspec import AbstractFileSystem
from torch.nn import Module, Parameter

from .._compat import has_safetensors
from .pytorch import ModuleIterator, apply_to_module

if TYPE_CHECKING:
    import safetensors

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

PathOrFileDescriptor = Union[str, IO]


class ModelFile(ABC):
    """
    Model files can be a local path or a remote path exposed as e.g. an I/O
    stream. This is a common base class for such different types of model
    files.
    """

    @abstractmethod
    def open(self, mode="rb", encoding=None) -> IO:
        """
        Get the model file as an I/O stream.

        :param mode:
            Mode to open the file with (see Python ``open``).
        :param encoding:
            Encoding to use when the file is opened as text.
        :returns:
            An I/O stream.
        """
        ...

    @property
    @abstractmethod
    def path(self) -> Optional[str]:
        """
        Get the model file as a local path. If the model file is not
        available as a local path, the value of this property is
        ``None``.
        """
        ...


class FsspecModelFile(ModelFile):
    """
    Model file on an fsspec filesystem.
    """

    def __init__(
        self,
        fs: AbstractFileSystem,
        path: str,
        fsspec_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Construct an fsspec model file representation.

        :param fs:
            The filesystem.
        :param path:
            The path of the model file on the filesystem.
        :param fsspec_args:
            Implementation-specific keyword arguments to pass to fsspec
            filesystem operations.
        """
        super().__init__()
        self._fs = fs
        self._path = path
        self._fsspec_args = fsspec_args

    def open(self, mode="rb", encoding=None) -> IO:
        return self._fs.open(
            self._path, mode=mode, encoding=encoding, **self._fsspec_args
        )

    @property
    def path(self) -> Optional[str]:
        return None


class LocalModelFile(ModelFile):
    """
    Model file on the local host machine.
    """

    def __init__(self, path: str):
        """
        Construct a local model file representation.

        :param path:
            The path of the model file on the local filesystem.
        """
        super().__init__()
        self._path = path

    def open(self, mode="rb", encoding=None) -> IO:
        return open(self._path, mode=mode, encoding=encoding)

    @property
    def path(self) -> Optional[str]:
        return self._path


class ModelCheckpointType(Enum):
    """
    Types of model checkpoints supported by Curated Transformers.
    """

    #: PyTorch `checkpoint<https://pytorch.org/docs/stable/generated/torch.save.html>`_.
    PYTORCH_STATE_DICT = 0

    #: Hugging Face `Safetensors <https://github.com/huggingface/safetensors>`_ checkpoint.
    SAFE_TENSORS = 1

    @property
    def loader(
        self,
    ) -> Callable[[Iterable[ModelFile]], Iterable[Mapping[str, torch.Tensor]]]:
        checkpoint_type_to_loader = {
            ModelCheckpointType.PYTORCH_STATE_DICT: _load_pytorch_state_dicts_from_checkpoints,
            ModelCheckpointType.SAFE_TENSORS: _load_safetensor_state_dicts_from_checkpoints,
        }
        return checkpoint_type_to_loader[self]

    @property
    def pretty_name(self) -> str:
        if self == ModelCheckpointType.PYTORCH_STATE_DICT:
            return "PyTorch StateDict"
        elif self == ModelCheckpointType.SAFE_TENSORS:
            return "SafeTensors"
        else:
            return ""


# When `None`, behaviour is implementation-specific.
_MODEL_CHECKPOINT_TYPE: ContextVar[Optional[ModelCheckpointType]] = ContextVar(
    "model_checkpoint_type", default=None
)


@contextmanager
def _use_model_checkpoint_type(
    model_checkpoint_type: ModelCheckpointType,
):
    """
    Specifies which type of model checkpoint to use when loading a serialized model.

    By default, Curated Transformers will attempt to load from the most suitable
    checkpoint type depending on its availability. This context manager can be used
    to override the default behaviour.

    .. code-block:: python

        with use_model_checkpoint_type(ModelCheckpointType.SAFETENSORS):
            encoder = BertEncoder.from_hf_hub(name="bert-base-uncased")
    """
    token = _MODEL_CHECKPOINT_TYPE.set(model_checkpoint_type)
    try:
        yield
    finally:
        _MODEL_CHECKPOINT_TYPE.reset(token)


def load_model_from_checkpoints(
    model: Module,
    *,
    filepaths: Iterable[ModelFile],
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
    return Parameter(tensor, requires_grad=old_param.requires_grad).to(device=device)  # type: ignore


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


def _load_safetensor_state_dicts_from_checkpoints(
    checkpoints: Iterable[ModelFile],
) -> Iterable[Mapping[str, torch.Tensor]]:
    if not has_safetensors:
        raise ValueError(
            "The `safetensors` library is required to load models from Safetensors checkpoints"
        )

    import safetensors.torch

    for checkpoint in checkpoints:
        # Prefer to load from a path when possible. Since loading from a file
        # temporarily puts the checkpoint in memory twice.
        if checkpoint.path is not None:
            # Map to CPU first to support all devices.
            state_dict = safetensors.torch.load_file(checkpoint.path, device="cpu")
        else:
            with checkpoint.open() as f:
                # This has memory overhead, since Safetensors does not have
                # support for loading from a file object and cannot use
                # the bytes in-place.
                checkpoint_bytes = f.read()
                state_dict = safetensors.torch.load(checkpoint_bytes)
        yield state_dict


def _load_pytorch_state_dicts_from_checkpoints(
    checkpoints: Iterable[ModelFile],
) -> Iterable[Mapping[str, torch.Tensor]]:
    for checkpoint in checkpoints:
        with checkpoint.open() as f:
            # Map to CPU first to support all devices.
            state_dict = torch.load(
                f, map_location=torch.device("cpu"), weights_only=True
            )
        yield state_dict
