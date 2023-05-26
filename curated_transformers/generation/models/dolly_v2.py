from typing import Iterator, List, Tuple, Type, TypeVar
import torch

from .generator_model import GeneratorModel
from ..greedy import Generator
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


class DollyV2Generator(GeneratorModel):
    def __init__(self, tokenizer: GPTNeoXTokenizer, causal_lm: GPTNeoXCausalLM):
        super().__init__()
        self.generator = StringGenerator(Generator(causal_lm), tokenizer)
        self.eos_id = tokenizer.processor.piece_id(END_KEY)

    @classmethod
    def from_hf_hub(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
        device: torch.device = torch.device("cuda")
    ) -> Self:
        tokenizer = GPTNeoXTokenizer.from_hf_hub(name=name, revision=revision)
        causal_lm = GPTNeoXCausalLM.from_hf_hub(name=name, revision=revision)
        causal_lm.to(device)
        return cls(tokenizer, causal_lm)

    def generate(self, prompts: List[str]) -> Iterator[Iterator[Tuple[int, str]]]:
        prompts_with_instructions = [
            to_prompt_with_instructions(prompt) for prompt in prompts
        ]
        yield from self.generator(prompts_with_instructions, eos_id=self.eos_id)


def to_prompt_with_instructions(prompt):
    return InputChunks(
        [
            TextChunk(INTRO_BLURB),
            SpecialPieceChunk(INSTRUCTION_KEY, before="\n\n", after="\n"),
            TextChunk(prompt),
            SpecialPieceChunk(RESPONSE_KEY, before="\n\n", after="\n"),
        ]
    )
