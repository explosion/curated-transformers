import dataclasses
from typing import Any, Generic, List, Optional, Type, TypeVar

import torch

from ..models.auto_model import AutoCausalLM
from ..models.module import CausalLMModule
from ..models.output import CacheT
from ..quantization.bnb.config import BitsAndBytesConfig
from ..tokenizers.auto_tokenizer import AutoTokenizer
from ..tokenizers.chunks import InputChunks, TextChunk
from ..tokenizers.tokenizer import TokenizerBase
from .config import GeneratorConfig, SampleGeneratorConfig
from .generator import Generator
from .generator_wrapper import GeneratorWrapper
from .hf_hub import FromHF
from .string_generator import StringGenerator

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="DefaultGenerator")


class DefaultGenerator(Generic[CacheT], GeneratorWrapper, FromHF):
    """
    Generator wrapper for models that do not need specific prompting.
    """

    def __init__(
        self,
        tokenizer: TokenizerBase,
        causal_lm: CausalLMModule[Any, CacheT],
        default_config: Optional[GeneratorConfig] = None,
    ):
        """
        Construct a generic generator.

        :param tokenizer:
            A tokenizer.
        :param causal_lm:
            A causal language model.
        :param default_config:
            Configuration to use as a default when the configuration provided
            to the ``generate`` method is underspecified. For instance, if the
            end-of-sequence identifier is ``None`` in the generation
            configuration, it will be taken from the default configuration.
        """
        super().__init__()

        eos_id = (
            tokenizer.piece_to_id(tokenizer.eos_piece)
            if tokenizer.eos_piece is not None
            else None
        )

        self.generator = StringGenerator(tokenizer, Generator(causal_lm))
        self.default_config = (
            SampleGeneratorConfig(max_generated_pieces=200, eos_id=eos_id)
            if default_config is None
            else default_config
        )

    @classmethod
    def from_hf_hub_to_cache(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
    ):
        AutoTokenizer.from_hf_hub_to_cache(name=name, revision=revision)
        AutoCausalLM.from_hf_hub_to_cache(name=name, revision=revision)

    @classmethod
    def from_hf_hub(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> Self:
        tokenizer = AutoTokenizer.from_hf_hub(name=name, revision=revision)
        causal_lm = AutoCausalLM.from_hf_hub(
            name=name,
            revision=revision,
            device=device,
            quantization_config=quantization_config,
        )
        return cls(tokenizer, causal_lm)

    def generate(self, prompts: List[str], config: GeneratorConfig) -> List[str]:
        eos_id = self.default_config.eos_id if config.eos_id is None else config.eos_id
        max_generated_pieces = (
            self.default_config.max_generated_pieces
            if config.max_generated_pieces is None
            else config.max_generated_pieces
        )
        config = dataclasses.replace(
            config, eos_id=eos_id, max_generated_pieces=max_generated_pieces
        )
        return self.generator(self.preprocess_prompts(prompts), config=config)

    def preprocess_prompts(self, prompts: List[str]) -> List[InputChunks]:
        """
        Prepare a list of prompts for generation.

        :param prompts:
            The prompts to prepare.
        :returns:
            Prepared prompts.
        """
        return [InputChunks([TextChunk(prompt)]) for prompt in prompts]
