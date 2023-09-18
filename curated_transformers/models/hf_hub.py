from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from fsspec import AbstractFileSystem
from torch import Tensor

from ..quantization import prepare_module_for_quantization
from ..quantization.bnb.config import BitsAndBytesConfig
from ..util.fsspec import (
    get_model_checkpoint_files as get_model_checkpoint_files_fsspec,
)
from ..util.fsspec import get_model_config as get_model_config_fsspec
from ..util.hf import get_model_checkpoint_files, get_model_config
from ..util.serde import ModelCheckpointType, ModelFile, load_model_from_checkpoints
from .module import TransformerModule

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FromHFHub")


class FromHFHub(ABC):
    """
    Mixin class for downloading models from Hugging Face Hub.

    A module using this mixin can implement the ``convert_hf_state_dict``
    and ``from_hf_config`` methods. The mixin will then provide the
    ``from_hf_hub`` method to download a model from the Hugging Face Hub.
    """

    @classmethod
    @abstractmethod
    def convert_hf_state_dict(
        cls, params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        """
        Convert a state dict of a Hugging Face model to a valid
        state dict for the module.

        :param params:
            The state dict to convert.
        :returns:
            The converted state dict.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        """
        Create the module from a Hugging Face model JSON-deserialized
        model configuration.

        :param hf_config:
            Hugging Face model configuration.
        :param device:
            Device on which to initialize the model.
        :returns:
            Module constructed using the configuration.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_hub_to_cache(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
    ):
        """
        Download the model's weights from Hugging Face Hub into the local
        Hugging Face cache directory. Subsequent loading of the
        model will read the weights from disk. If the weights are already
        cached, this is a no-op.

        :param name:
            Model name.
        :param revision:
            Model revision.
        """
        _ = get_model_config(name, revision)
        _ = get_model_checkpoint_files(name, revision)

    @classmethod
    def from_fsspec(
        cls: Type[Self],
        *,
        fs: AbstractFileSystem,
        model_path: str,
        fsspec_args: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> Self:
        """
        Construct a module and load its parameters from a fsspec filesystem.

        :param fs:
            The filesystem to load the model from.
        :param model_path:
            The path of the model on the filesystem.
        :param fsspec_args:
            Implementation-specific keyword arguments to pass to fsspec
            filesystem operations.
        :param device:
            Device on which the model is initialized.
        :param quantization_config:
            Configuration for loading quantized weights.
        :returns:
            Module with the parameters loaded.
        """
        return cls._create_and_load_model(
            get_config=lambda: get_model_config_fsspec(
                fs, model_path, fsspec_args=fsspec_args
            ),
            get_checkpoint_files=lambda: get_model_checkpoint_files_fsspec(
                fs, model_path, fsspec_args=fsspec_args
            ),
            device=device,
            quantization_config=quantization_config,
        )

    @classmethod
    def from_hf_hub(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> Self:
        """
        Construct a module and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :param device:
            Device on which the model is initialized.
        :param quantization_config:
            Configuration for loading quantized weights.
        :returns:
            Module with the parameters loaded.
        """
        return cls._create_and_load_model(
            get_config=lambda: get_model_config(name, revision),
            get_checkpoint_files=lambda: get_model_checkpoint_files(name, revision),
            device=device,
            quantization_config=quantization_config,
        )

    @abstractmethod
    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ):
        """
        Moves and/or casts the parameters and buffers.

        This method is automatically implemented by also deriving from
        ``torch.nn.Module``. This mixin does not derive from ``Module`` in
        order to be an abstract base class.
        """
        ...

    @classmethod
    def _create_and_load_model(
        cls: Type[Self],
        *,
        get_config: Callable[[], Dict[Any, str]],
        get_checkpoint_files: Callable[[], Tuple[List[ModelFile], ModelCheckpointType]],
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> Self:
        config = get_config()
        model = cls.from_hf_config(hf_config=config, device=torch.device("meta"))

        # Convert the model to the expected dtype.
        assert isinstance(model, TransformerModule)
        dtype: torch.dtype = model.config.dtype
        serialized_dtype_str = config.get("torch_dtype")
        if serialized_dtype_str is not None:
            serialized_dtype = getattr(torch, serialized_dtype_str, None)
            if not isinstance(serialized_dtype, torch.dtype):
                raise ValueError(f"Invalid torch dtype `{serialized_dtype_str}`")
            dtype = serialized_dtype
        model.to(dtype=dtype)

        # Prepare for quantization.
        if quantization_config is not None:
            tensor2param = prepare_module_for_quantization(model, quantization_config)  # type: ignore
        else:
            tensor2param = None

        # Download model and convert HF parameter names to ours.
        checkpoint_filenames, checkpoint_type = get_checkpoint_files()
        load_model_from_checkpoints(
            model,  # type:ignore
            filepaths=checkpoint_filenames,
            checkpoint_type=checkpoint_type,
            state_dict_converter=cls.convert_hf_state_dict,
            tensor_to_param_converter=tensor2param,
            device=device,
        )

        # Ensure that any non-persistent buffers are also moved to
        # the correct device.
        if device is not None:
            model.to(device)

        return model


def _process_hf_keys(
    model_name: str,
    hf_config: Dict[str, Any],
    hf_to_curated: Dict[str, Union[str, Tuple[str, Callable]]],
    extra_keys: List[str] = [],
) -> Dict[str, Any]:
    """
    Convert Hugging Face configuration keys to keyword arguments for
    Curated Transformers configuration classes.

    :param model_name:
        Model name. Only used in exception messages.
    :param hf_config:
        Hugging Face model configuration.
    :param hf_to_curated:
        Dictionay that maps Hugging Face configuration keys to keyword
        arguments for a Curated Transformers configuration class. If a value
        is a tuple, the first tuple element is the name of the keyword
        argument class and the second tuple element is a conversion function.
    :param extra_keys:
        Optional keys for which the Hugging Face configuration key and the
        keyword argument of the Curated Transformers configuration class is
        the same.
    :returns:
        Dictionary with keyword arguments.
    """
    missing_keys = tuple(
        sorted(set(hf_to_curated.keys()).difference(set(hf_config.keys())))
    )
    if len(missing_keys) != 0:
        raise ValueError(
            f"Missing keys in Hugging Face {model_name} model config: {missing_keys}"
        )

    kwargs = {}

    for hf, curated in hf_to_curated.items():
        if isinstance(curated, tuple):
            curated, ctor = curated
        else:
            ctor = lambda x: x

        kwargs[curated] = ctor(hf_config[hf])

    # Handle config options that are not set in all models.
    kwargs.update({k: hf_config[k] for k in extra_keys if k in hf_config})

    return kwargs
