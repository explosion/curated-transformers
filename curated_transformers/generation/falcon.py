from typing import List, TypeVar

from ..models.falcon import FalconCausalLM
from ..tokenizers.chunks import InputChunks, TextChunk
from ..tokenizers.tokenizer import Tokenizer
from .default_generator import DefaultGenerator
from .hf_hub import FromHF

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconGenerator")


class FalconGenerator(DefaultGenerator, FromHF):
    """
    Generator for Falcon model variants.
    """

    def __init__(self, tokenizer: Tokenizer, causal_lm: FalconCausalLM):
        """
        Construct a Falcon generator.

        :param tokenizer:
            A Falcon tokenizer.
        :param causal_lm:
            A Falcon causal language model.

        """
        super().__init__(tokenizer, causal_lm)

    def preprocess_prompts(self, prompts: List[str]) -> List[InputChunks]:
        return [
            InputChunks(
                [
                    TextChunk(prompt.strip()),
                    # Add a newline, otherwise Falcon will generate an answer that
                    # starts with a newline.
                    TextChunk("\n"),
                ]
            )
            for prompt in prompts
        ]
