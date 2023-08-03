from typing import Dict, Optional, Type, cast

import torch

from ..models.auto_model import AutoModel
from ..quantization.bnb.config import BitsAndBytesConfig
from .default_generator import DefaultGenerator
from .dolly_v2 import DollyV2Generator
from .falcon import FalconGenerator
from .generator_wrapper import GeneratorWrapper
from .hf_hub import FromHFHub

# For the time being, we enable support for a generator on a case-by-case basis.
# In the future we might defer all unknown generators to DefaultGenerator.
GENERATOR_MAP: Dict[str, Type[FromHFHub]] = {
    "dolly-v2": DollyV2Generator,
    "falcon": FalconGenerator,
    "llama": DefaultGenerator,
}


class AutoGenerator(AutoModel[GeneratorWrapper]):
    """Causal LM generator loaded from the Hugging Face Model Hub.

    .. attention::
            This class can currently only be used with the following models:

            - Models based on Dolly v2 (contain ``dolly-v2`` in the name).
            - Models based on Falcon (contain ``falcon`` in the name).
            - Models based on Llama (contain ``llama`` in the name).
    """

    @classmethod
    def from_hf_hub(
        cls,
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> GeneratorWrapper:
        # We need to match the name of the model directly as our
        # generators bundle model-specific prompts that are specific
        # to certain fine-tuned models.
        generator: Optional[GeneratorWrapper] = None
        for substring, generator_cls in GENERATOR_MAP.items():
            if substring in name.lower():
                generator = cast(
                    GeneratorWrapper,
                    generator_cls.from_hf_hub(
                        name=name,
                        revision=revision,
                        device=device,
                        quantization_config=quantization_config,
                    ),
                )
                break

        if generator is None:
            supported_models = list(sorted(GENERATOR_MAP.keys()))
            raise ValueError(
                f"Unsupported generator model `{name}` for {cls.__name__}. "
                f"Supported model variants: `{supported_models}`"
            )

        assert isinstance(generator, GeneratorWrapper)
        return generator
