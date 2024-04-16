import warnings
from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar

import torch
from catalogue import Registry
from fsspec import AbstractFileSystem

from curated_transformers import registry

from ..layers.cache import KeyValueCache
from ..quantization.bnb.config import BitsAndBytesConfig
from ..repository.fsspec import FsspecArgs, FsspecRepository
from ..repository.hf_hub import HfHubRepository
from ..repository.repository import ModelRepository, Repository
from .config import TransformerConfig
from .hf_hub import FromHF
from .module import CausalLMModule, DecoderModule, EncoderModule, TransformerModule

ModelT = TypeVar("ModelT")


class AutoModel(ABC, Generic[ModelT]):
    """
    Base class for models that can be loaded from the Hugging
    Face Model Hub.
    """

    _base_cls: Type[TransformerModule]
    _registry: Registry

    @classmethod
    def _resolve_model_cls(
        cls,
        repo: ModelRepository,
    ) -> Type[FromHF]:
        config = repo.model_config()

        for entrypoint, module_cls in cls._registry.get_entry_points().items():
            if not issubclass(module_cls, FromHF):
                warnings.warn(
                    f"Entry point `{entrypoint}` cannot load from Hugging Face Hub "
                    "since the FromHF mixin is not implemented"
                )
                continue

            if not issubclass(module_cls, cls._base_cls):
                warnings.warn(
                    f"Entry point `{entrypoint}` cannot be used by `{cls.__name__}` "
                    f"since it does does not have `{cls._base_cls.__name__}` "
                    "as its base class"
                )
                continue

            if module_cls.is_supported(config):
                return module_cls

        entrypoints = {
            entrypoint for entrypoint in cls._registry.get_entry_points().keys()
        }

        raise ValueError(
            f"Unsupported model type for `{cls.__name__}`. "
            f"Registered models: {', '.join(sorted(entrypoints))}"
        )

    @classmethod
    def _instantiate_model(
        cls,
        repo: Repository,
        device: Optional[torch.device],
        quantization_config: Optional[BitsAndBytesConfig],
    ) -> FromHF:
        module_cls = cls._resolve_model_cls(ModelRepository(repo))
        module = module_cls.from_repo(
            repo=repo,
            device=device,
            quantization_config=quantization_config,
        )
        return module

    @classmethod
    def from_fsspec(
        cls,
        *,
        fs: AbstractFileSystem,
        model_path: str,
        fsspec_args: Optional[FsspecArgs] = None,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> ModelT:
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
            repo=FsspecRepository(
                fs,
                path=model_path,
                fsspec_args=fsspec_args,
            ),
            device=device,
            quantization_config=quantization_config,
        )

    @classmethod
    def from_hf_hub(
        cls,
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> ModelT:
        """
        Construct and load a model or a generator from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :param device:
            Device on which to initialize the model.
        :param quantization_config:
            Configuration for loading quantized weights.
        :returns:
            Loaded model or generator.
        """
        return cls.from_repo(
            repo=HfHubRepository(name=name, revision=revision),
            device=device,
            quantization_config=quantization_config,
        )

    @classmethod
    @abstractmethod
    def from_repo(
        cls,
        *,
        repo: Repository,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> ModelT:
        """
        Construct and load a model or a generator from a repository.

        :param repository:
            The repository to load from.
        :param device:
            Device on which to initialize the model.
        :param quantization_config:
            Configuration for loading quantized weights.
        :returns:
            Loaded model or generator.
        """

        raise NotImplementedError

    @classmethod
    def from_hf_hub_to_cache(
        cls,
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


class AutoEncoder(AutoModel[EncoderModule[TransformerConfig]]):
    """
    Encoder model loaded from the Hugging Face Model Hub.
    """

    _base_cls = EncoderModule
    _registry: Registry = registry.encoders

    @classmethod
    def from_repo(
        cls,
        *,
        repo: Repository,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> EncoderModule[TransformerConfig]:
        encoder = cls._instantiate_model(repo, device, quantization_config)
        assert isinstance(encoder, EncoderModule)
        return encoder


class AutoDecoder(AutoModel[DecoderModule[TransformerConfig, KeyValueCache]]):
    """
    Decoder module loaded from the Hugging Face Model Hub.
    """

    _base_cls = DecoderModule
    _registry = registry.decoders

    @classmethod
    def from_repo(
        cls,
        *,
        repo: Repository,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> DecoderModule[TransformerConfig, KeyValueCache]:
        decoder = cls._instantiate_model(repo, device, quantization_config)
        assert isinstance(decoder, DecoderModule)
        return decoder


class AutoCausalLM(AutoModel[CausalLMModule[TransformerConfig, KeyValueCache]]):
    """
    Causal LM model loaded from the Hugging Face Model Hub.
    """

    _base_cls = CausalLMModule
    _registry: Registry = registry.causal_lms

    @classmethod
    def from_repo(
        cls,
        *,
        repo: Repository,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> CausalLMModule[TransformerConfig, KeyValueCache]:
        causal_lm = cls._instantiate_model(repo, device, quantization_config)
        assert isinstance(causal_lm, CausalLMModule)
        return causal_lm
