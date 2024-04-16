from typing import TypeVar

from ..models.llama import LlamaCausalLM
from ..tokenizers.tokenizer import Tokenizer
from .default_generator import DefaultGenerator
from .hf_hub import FromHF

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="LlamaGenerator")


class LlamaGenerator(DefaultGenerator, FromHF):
    """
    Generator for Llama and Llama 2 model variants.
    """

    def __init__(self, tokenizer: Tokenizer, causal_lm: LlamaCausalLM):
        """
        Construct a Llama generator.

        :param tokenizer:
            A Llama tokenizer.
        :param causal_lm:
            A Llama causal language model.

        """
        super().__init__(
            tokenizer,
            causal_lm,
        )
