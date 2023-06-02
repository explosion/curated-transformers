from typing import Iterator, List, Optional, Tuple, Type, TypeVar
import torch

from .generation_model import GenerationModel
from ..generator import Generator
from ...models.gpt_neox.causal_lm import GPTNeoXCausalLM
from ..string_generator import StringGenerator
from ...tokenization.chunks import InputChunks, SpecialPieceChunk, TextChunk
from ...tokenization.gpt_neox_tokenizer import GPTNeoXTokenizer

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="DollyV2Generator")


class DollyV2Generator(GenerationModel):
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

    def generate(self, prompts: List[str]) -> Iterator[List[Tuple[int, str]]]:
        prompts_with_instructions = [
            _to_prompt_with_instructions(prompt) for prompt in prompts
        ]
        yield from self.generator(prompts_with_instructions, eos_id=self.eos_id)


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
