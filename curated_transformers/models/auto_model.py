from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, Type, TypeVar

import torch
from fsspec import AbstractFileSystem

from ..layers.cache import KeyValueCache
from ..quantization.bnb.config import BitsAndBytesConfig
from ..repository.fsspec import FsspecArgs, FsspecRepository
from ..repository.hf_hub import HfHubRepository
from ..repository.repository import ModelRepository, Repository
from .albert import ALBERTEncoder
from .bert import BERTEncoder
from .camembert import CamemBERTEncoder
from .config import TransformerConfig
from .falcon import FalconCausalLM, FalconDecoder
from .gpt_neox import GPTNeoXCausalLM, GPTNeoXDecoder
from .hf_hub import FromHFHub
from .llama import LlamaCausalLM, LlamaDecoder
from .module import CausalLMModule, DecoderModule, EncoderModule
from .mpt.causal_lm import MPTCausalLM
from .mpt.decoder import MPTDecoder
from .roberta import RoBERTaEncoder
from .xlm_roberta import XLMREncoder

ModelT = TypeVar("ModelT")


class AutoModel(ABC, Generic[ModelT]):
    """
    Base class for models that can be loaded from the Hugging
    Face Model Hub.
    """

    _hf_model_type_to_curated: Dict[str, Type[FromHFHub]] = {}

    @classmethod
    def _resolve_model_cls(
        cls,
        repo: ModelRepository,
    ) -> Type[FromHFHub]:
        model_type = repo.model_type()
        module_cls = cls._hf_model_type_to_curated.get(model_type)
        if module_cls is None:
            raise ValueError(
                f"Unsupported model type `{model_type}` for {cls.__name__}. "
                f"Supported model types: {tuple(cls._hf_model_type_to_curated.keys())}"
            )
        assert issubclass(module_cls, FromHFHub)
        return module_cls

    @classmethod
    def _instantiate_model(
        cls,
        repo: Repository,
        device: Optional[torch.device],
        quantization_config: Optional[BitsAndBytesConfig],
    ) -> FromHFHub:
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

    _hf_model_type_to_curated: Dict[str, Type[FromHFHub]] = {
        "bert": BERTEncoder,
        "albert": ALBERTEncoder,
        "camembert": CamemBERTEncoder,
        "roberta": RoBERTaEncoder,
        "xlm-roberta": XLMREncoder,
    }

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

    _hf_model_type_to_curated: Dict[str, Type[FromHFHub]] = {
        "falcon": FalconDecoder,
        "gpt_neox": GPTNeoXDecoder,
        "llama": LlamaDecoder,
        "mpt": MPTDecoder,
        "RefinedWeb": FalconDecoder,
        "RefinedWebModel": FalconDecoder,
    }

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

    _hf_model_type_to_curated: Dict[str, Type[FromHFHub]] = {
        "falcon": FalconCausalLM,
        "gpt_neox": GPTNeoXCausalLM,
        "llama": LlamaCausalLM,
        "mpt": MPTCausalLM,
        "RefinedWeb": FalconCausalLM,
        "RefinedWebModel": FalconCausalLM,
    }

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
