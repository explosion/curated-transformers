import json
from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import torch

from ..generation.dolly_v2 import DollyV2Generator
from ..generation.falcon import FalconGenerator
from ..generation.generator_wrapper import GeneratorWrapper
from ..models.albert import AlbertEncoder
from ..models.bert import BertEncoder
from ..models.camembert import CamembertEncoder
from ..models.gpt_neox import GPTNeoXCausalLM, GPTNeoXDecoder
from ..models.hf_hub import FromPretrainedHFModel
from ..models.module import CausalLMModule, DecoderModule, EncoderModule
from ..models.output import KeyValueCache
from ..models.refined_web_model import RefinedWebModelCausalLM, RefinedWebModelDecoder
from ..models.roberta import RobertaEncoder
from ..models.xlm_roberta import XlmRobertaEncoder
from ..quantization import BitsAndBytesConfig
from .hf import get_model_config_filepath

ModelT = TypeVar("ModelT")


class AutoModel(ABC, Generic[ModelT]):
    """Base class for models that can be loaded from the Hugging
    Face Model Hub."""

    @classmethod
    @abstractmethod
    def from_hf_hub(
        cls,
        name: str,
        revision: str = "main",
        *,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> ModelT:
        """Construct and load a module or a generator from Hugging Face Hub.

        :param name:
            Model name.
        :param revsion:
            Model revision.
        :param device:
            Device on which to initialize the model.
        :param quantization_config:
            Configuration for loading quantized weights.
        :returns:
            Loaded module or generator.
        """
        raise NotImplementedError


class AutoEncoder(AutoModel[EncoderModule]):
    """Encoder module loaded from the Hugging Face Model Hub."""

    _HF_MODEL_TYPE_TO_CURATED = {
        "bert": BertEncoder,
        "albert": AlbertEncoder,
        "camembert": CamembertEncoder,
        "roberta": RobertaEncoder,
        "xlm-roberta": XlmRobertaEncoder,
    }

    @classmethod
    def from_hf_hub(
        cls,
        name: str,
        revision: str = "main",
        *,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> EncoderModule:
        encoder = _instantiate_module_from_hf_hub(
            name,
            revision,
            device,
            quantization_config,
            cls._HF_MODEL_TYPE_TO_CURATED,
            "encoder",
        )
        assert isinstance(encoder, EncoderModule)
        return encoder


class AutoDecoder(AutoModel[DecoderModule]):
    """Decoder module loaded from the Hugging Face Model Hub."""

    _HF_MODEL_TYPE_TO_CURATED = {
        "gpt_neox": GPTNeoXDecoder,
        "RefinedWeb": RefinedWebModelDecoder,
        "RefinedWebModel": RefinedWebModelDecoder,
    }

    @classmethod
    def from_hf_hub(
        cls,
        name: str,
        revision: str = "main",
        *,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> DecoderModule:
        decoder = _instantiate_module_from_hf_hub(
            name,
            revision,
            device,
            quantization_config,
            cls._HF_MODEL_TYPE_TO_CURATED,
            "decoder",
        )
        assert isinstance(decoder, DecoderModule)
        return decoder


class AutoCausalLM(AutoModel[CausalLMModule[KeyValueCache]]):
    """Causal LM module loaded from the Hugging Face Model Hub."""

    _HF_MODEL_TYPE_TO_CURATED = {
        "gpt_neox": GPTNeoXCausalLM,
        "RefinedWeb": RefinedWebModelCausalLM,
        "RefinedWebModel": RefinedWebModelCausalLM,
    }

    @classmethod
    def from_hf_hub(
        cls,
        name: str,
        revision: str = "main",
        *,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> CausalLMModule[KeyValueCache]:
        causal_lm = _instantiate_module_from_hf_hub(
            name,
            revision,
            device,
            quantization_config,
            cls._HF_MODEL_TYPE_TO_CURATED,
            "causal LM",
        )
        assert isinstance(causal_lm, CausalLMModule)
        return causal_lm


class AutoGenerator(AutoModel[GeneratorWrapper]):
    """Causal LM generator loaded from the Hugging Face Model Hub.

    **NOTE** - This class can currently only be used with following models:
        - ``databricks/dolly-v2`` variants.
        - ``tiiuae/falcon`` variants.
    """

    _DOLLY_V2_PREFIX = "databricks/dolly-v2"
    _FALCON_PREFIX = "tiiuae/falcon"

    @classmethod
    def from_hf_hub(
        cls,
        name: str,
        revision: str = "main",
        *,
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> GeneratorWrapper:
        # We need to match the name of the model directly as our
        # generators bundle model-specific prompts that are specific
        # to certain fine-tuned models.
        generator: Optional[GeneratorWrapper] = None
        if name.startswith(cls._DOLLY_V2_PREFIX):
            generator = DollyV2Generator.from_hf_hub(
                name=name,
                revision=revision,
                device=device,
                quantization_config=quantization_config,
            )
        elif name.startswith(cls._FALCON_PREFIX):
            generator = FalconGenerator.from_hf_hub(
                name=name,
                revision=revision,
                device=device,
                quantization_config=quantization_config,
            )
        else:
            supported_models = (cls._DOLLY_V2_PREFIX, cls._FALCON_PREFIX)
            raise ValueError(
                f"Unsupported generator model `{name}`. "
                f"Supported model: `{supported_models}` variants"
            )

        assert generator is not None
        assert isinstance(generator, GeneratorWrapper)
        return generator


def _instantiate_module_from_hf_hub(
    name: str,
    revision: str,
    device: Optional[torch.device],
    quantization_config: Optional[BitsAndBytesConfig],
    model_type_to_class_map: Mapping[str, Type],
    module_type: str,
) -> FromPretrainedHFModel:
    model_type = _get_hf_config_model_type(name, revision)
    module_cls = model_type_to_class_map.get(model_type)
    if module_cls is None:
        raise ValueError(
            f"Unsupported model type `{model_type}` for {module_type}. "
            f"Supported model types: {tuple(model_type_to_class_map.keys())}"
        )
    assert issubclass(module_cls, FromPretrainedHFModel)
    module = module_cls.from_hf_hub(
        name, revision, device=device, quantization_config=quantization_config
    )
    return module


def _get_hf_config_model_type(name: str, revision: str) -> Any:
    config_filename = get_model_config_filepath(name, revision)
    with open(config_filename, "r") as f:
        config = json.load(f)
        model_type = config.get("model_type")
        if model_type is None:
            raise ValueError("Model type not found in Hugging Face model config")
        return model_type
