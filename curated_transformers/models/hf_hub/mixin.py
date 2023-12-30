from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Mapping, Optional, Type, TypeVar

import torch
from fsspec import AbstractFileSystem
from torch import Tensor
from torch.nn import Module

from ...quantization import prepare_module_for_quantization
from ...quantization.bnb.config import BitsAndBytesConfig
from ...repository.fsspec import FsspecArgs, FsspecRepository
from ...repository.hf_hub import HfHubRepository
from ...repository.repository import ModelRepository, Repository
from ...util.serde.load import load_model_from_checkpoints
from ..module import ConfigT, TransformerModule

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FromHFHub")


class FromHFHub(ABC, Generic[ConfigT]):
    """
    Mixin class for downloading models from Hugging Face Hub.

    A module using this mixin can implement the ``convert_hf_state_dict``
    and ``from_hf_config`` methods. The mixin will then provide the
    ``from_hf_hub`` method to download a model from the Hugging Face Hub.
    """

    @classmethod
    def convert_hf_state_dict(
        cls, params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        """
        Alias for :meth:`.state_dict_from_hf`.
        """
        return cls.state_dict_from_hf(params)

    @classmethod
    @abstractmethod
    def state_dict_from_hf(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        Convert a state dict of a Hugging Face model to a valid
        state dict for the module.

        :param params:
            The state dict to convert.
        :returns:
            The converted state dict.
        """
        ...

    @classmethod
    @abstractmethod
    def state_dict_to_hf(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        Convert the state dict of the module to a compatible
        Hugging Face model's format.

        :param params:
            The state dict to convert.
        :returns:
            The converted state dict.
        """
        ...

    @classmethod
    @abstractmethod
    def config_from_hf(cls, hf_config: Mapping[str, Any]) -> ConfigT:
        """
        Convert a Hugging Face model configuration to the
        module's configuration.

        :param hf_config:
            The Hugging Face model configuration.
        :returns:
            The converted Curated Transformer
            configuration.
        """
        ...

    @classmethod
    @abstractmethod
    def config_to_hf(cls, curated_config: ConfigT) -> Mapping[str, Any]:
        """
        Convert the module's  configuration to the a
        Hugging Face model configuration.

        :param curated_config:
            The Curated Transformer model configuration.
        :returns:
            The converted Hugging Face configuration.
        """
        ...

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
        ...

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
        repo = ModelRepository(HfHubRepository(name=name, revision=revision))
        repo.model_config()
        repo.model_checkpoints()

    @classmethod
    def from_fsspec(
        cls: Type[Self],
        *,
        fs: AbstractFileSystem,
        model_path: str,
        fsspec_args: Optional[FsspecArgs] = None,
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
        return cls.from_repo(
            repo=FsspecRepository(fs, model_path, fsspec_args),
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
        return cls.from_repo(
            repo=HfHubRepository(name=name, revision=revision),
            device=device,
            quantization_config=quantization_config,
        )

    @classmethod
    @abstractmethod
    def is_supported(cls: Type[Self], config: Dict[str, Any]) -> bool:
        """
        Check if the model with the given configuration is supported by this
        class.

        :param config:
            Hugging Face model configuration.
        :returns:
            Whether the model is supported by this class.
        """
        raise NotImplementedError

    @classmethod
    def from_repo(
        cls: Type[Self],
        *,
        repo: Repository,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> Self:
        """
        Construct and load a model from a repository.

        :param repository:
            The repository to load from.
        :param device:
            Device on which to initialize the model.
        :param quantization_config:
            Configuration for loading quantized weights.
        :returns:
            Loaded model.
        """
        model_repo = ModelRepository(repo)
        config = model_repo.model_config()
        model = cls.from_hf_config(hf_config=config, device=torch.device("meta"))
        assert isinstance(model, Module)

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
        checkpoint_filenames, checkpoint_type = model_repo.model_checkpoints()
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
