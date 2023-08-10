from typing import TypeVar

from ..models.llama import LlamaCausalLM
from ..tokenizers.tokenizer import Tokenizer
from .default_generator import DefaultGenerator
from .hf_hub import FromHFHub


class MPTGenerator(DefaultGenerator, FromHFHub):
    """
    Generator for MPT model variants.
    """

    def __init__(self, tokenizer: Tokenizer, causal_lm: LlamaCausalLM):
        """
        Construct an MPT generator.

        :param tokenizer:
            An MPT tokenizer.
        :param causal_lm:
            An MPT causal language model.
        """
        super().__init__(
            tokenizer,
            causal_lm,
        )
