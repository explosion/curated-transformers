from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, Type, TypeVar

import torch

from ..layers.cache import KeyValueCache
from ..quantization.bnb.config import BitsAndBytesConfig
from ..util.hf import get_hf_config_model_type
from .albert import ALBERTEncoder
from .bert import BERTEncoder
from .camembert import CamemBERTEncoder
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
        name: str,
        revision: str,
    ) -> Type[FromHFHub]:
        model_type = get_hf_config_model_type(name, revision)
        module_cls = cls._hf_model_type_to_curated.get(model_type)
        if module_cls is None:
            raise ValueError(
                f"Unsupported model type `{model_type}` for {cls.__name__}. "
                f"Supported model types: {tuple(cls._hf_model_type_to_curated.keys())}"
            )
        assert issubclass(module_cls, FromHFHub)
        return module_cls

    @classmethod
    def _instantiate_model_from_hf_hub(
        cls,
        name: str,
        revision: str,
        device: Optional[torch.device],
        quantization_config: Optional[BitsAndBytesConfig],
    ) -> FromHFHub:
        module_cls = cls._resolve_model_cls(name, revision)
        module = module_cls.from_hf_hub(
            name=name,
            revision=revision,
            device=device,
            quantization_config=quantization_config,
        )
        return module

    @classmethod
    @abstractmethod
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
        module_cls = cls._resolve_model_cls(name, revision)
        module_cls.from_hf_hub_to_cache(name=name, revision=revision)


class AutoEncoder(AutoModel[EncoderModule]):
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
    def from_hf_hub(
        cls,
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> EncoderModule:
        encoder = cls._instantiate_model_from_hf_hub(
            name, revision, device, quantization_config
        )
        assert isinstance(encoder, EncoderModule)
        return encoder


class AutoDecoder(AutoModel[DecoderModule]):
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
    def from_hf_hub(
        cls,
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> DecoderModule:
        decoder = cls._instantiate_model_from_hf_hub(
            name, revision, device, quantization_config
        )
        assert isinstance(decoder, DecoderModule)
        return decoder


class AutoCausalLM(AutoModel[CausalLMModule[KeyValueCache]]):
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
    def from_hf_hub(
        cls,
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> CausalLMModule[KeyValueCache]:
        causal_lm = cls._instantiate_model_from_hf_hub(
            name, revision, device, quantization_config
        )
        assert isinstance(causal_lm, CausalLMModule)
        return causal_lm
