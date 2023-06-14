import dataclasses
from typing import List, Optional, Type, TypeVar

import torch

from ..models.gpt_neox.causal_lm import GPTNeoXCausalLM
from ..tokenization.chunks import InputChunks, SpecialPieceChunk, TextChunk
from ..tokenization.gpt_neox_tokenizer import GPTNeoXTokenizer
from .config import GeneratorConfig
from .generator import Generator
from .generator_wrapper import GeneratorWrapper
from .hf_hub import FromHFHub
from .string_generator import StringGenerator

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="DollyV2Generator")


class DollyV2Generator(GeneratorWrapper, FromHFHub):
    """Dolly v2 generator."""

    def __init__(self, tokenizer: GPTNeoXTokenizer, causal_lm: GPTNeoXCausalLM):
        """Construct a Dolly v2 generator.

        :param tokenizer:
            A Dolly v2 tokenizer.
        :param causal_lm:
            A Dolly v2 causal language model.

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
        device: Optional[torch.device] = None
    ) -> Self:
        tokenizer = GPTNeoXTokenizer.from_hf_hub(name=name, revision=revision)
        causal_lm = GPTNeoXCausalLM.from_hf_hub(
            name=name, revision=revision, device=device
        )
        return cls(tokenizer, causal_lm)

    def generate(self, prompts: List[str], config: GeneratorConfig) -> List[str]:
        # Fill config when necessary.
        eos_id = self.eos_id if config.eos_id is None else config.eos_id
        max_generated_pieces = (
            256 if config.max_generated_pieces is None else config.max_generated_pieces
        )
        config = dataclasses.replace(
            config, eos_id=eos_id, max_generated_pieces=max_generated_pieces
        )

        prompts_with_instructions = [
            _to_prompt_with_instructions(prompt) for prompt in prompts
        ]
        return self.generator(
            prompts_with_instructions, eos_id=self.eos_id, config=config
        )


def _to_prompt_with_instructions(prompt):
    return InputChunks(
        [
            TextChunk(INTRO_BLURB),
            # Dolly is really picky about getting the correct number of line
            # breaks, so the double line breaks in this chunk and the last
            # chunk are intentional.
            SpecialPieceChunk(INSTRUCTION_KEY, before="\n\n", after="\n"),
            TextChunk(prompt),
            SpecialPieceChunk(RESPONSE_KEY, before="\n\n", after="\n"),
        ]
    )
