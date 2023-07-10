from typing import List, TypeVar

from ..tokenizers.chunks import InputChunks, TextChunk
from .default_generator import DefaultGenerator
from .hf_hub import FromHFHub

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconGenerator")


class FalconGenerator(DefaultGenerator, FromHFHub):
    """
    Generator for Falcon model variants.
    """

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
