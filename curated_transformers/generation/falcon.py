import dataclasses
from typing import List, Optional, Type, TypeVar

import torch

from ..models.refined_web_model.causal_lm import RefinedWebModelCausalLM
from ..quantization.bnb.config import BitsAndBytesConfig
from ..tokenization.chunks import InputChunks, TextChunk
from ..tokenization.gpt_neox_tokenizer import GPTNeoXTokenizer
from .config import GeneratorConfig
from .generator import Generator
from .generator_wrapper import GeneratorWrapper
from .hf_hub import FromHFHub
from .string_generator import StringGenerator

END_KEY = "<|endoftext|>"


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconGenerator")


class FalconGenerator(GeneratorWrapper, FromHFHub):
    """Falcon generator."""

    def __init__(self, tokenizer: GPTNeoXTokenizer, causal_lm: RefinedWebModelCausalLM):
        """Construct a Falcon generator.

        :param tokenizer:
            A Falcon tokenizer.
        :param causal_lm:
            A Falcon causal language model.

        """
        super().__init__()
        self.generator = StringGenerator(tokenizer, Generator(causal_lm))
        self.eos_id = tokenizer.processor.piece_id(END_KEY)

    @classmethod
    def from_hf_hub(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> Self:
        tokenizer = GPTNeoXTokenizer.from_hf_hub(name=name, revision=revision)
        causal_lm = RefinedWebModelCausalLM.from_hf_hub(
            name=name,
            revision=revision,
            device=device,
            quantization_config=quantization_config,
        )
        return cls(tokenizer, causal_lm)

    def generate(self, prompts: List[str], config: GeneratorConfig) -> List[str]:
        # Fill config when necessary.
        eos_id = self.eos_id if config.eos_id is None else config.eos_id
        max_generated_pieces = (
            200 if config.max_generated_pieces is None else config.max_generated_pieces
        )
        config = dataclasses.replace(
            config, eos_id=eos_id, max_generated_pieces=max_generated_pieces
        )

        preprocessed_prompts = [_preprocess_prompt(prompt) for prompt in prompts]
        return self.generator(preprocessed_prompts, config=config)


def _preprocess_prompt(prompt):
    return InputChunks(
        [
            TextChunk(prompt.strip()),
            # Add a newline, otherwise Falcon will generate an answer that
            # starts with a newline.
            TextChunk("\n"),
        ]
    )
