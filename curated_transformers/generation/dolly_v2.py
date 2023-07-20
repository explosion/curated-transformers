from typing import List, TypeVar

from ..models.gpt_neox.causal_lm import GPTNeoXCausalLM
from ..tokenizers.chunks import InputChunks, SpecialPieceChunk, TextChunk
from ..tokenizers.tokenizer import Tokenizer
from .config import SampleGeneratorConfig
from .default_generator import DefaultGenerator

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="DollyV2Generator")


class DollyV2Generator(DefaultGenerator):
    """
    Generator for Dolly v2 model variants.
    """

    def __init__(self, tokenizer: Tokenizer, causal_lm: GPTNeoXCausalLM):
        """
        Construct a Dolly v2 generator.

        :param tokenizer:
            A Dolly v2 tokenizer.
        :param causal_lm:
            A Dolly v2 causal language model.

        """
        eos_id = tokenizer.tokenizer.token_to_id(END_KEY)
        super().__init__(
            tokenizer,
            causal_lm,
            default_config=SampleGeneratorConfig(
                eos_id=eos_id,
                max_generated_pieces=256,
                top_p=0.92,
            ),
        )

    def preprocess_prompts(self, prompts: List[str]) -> List[InputChunks]:
        return [
            InputChunks(
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
            for prompt in prompts
        ]
