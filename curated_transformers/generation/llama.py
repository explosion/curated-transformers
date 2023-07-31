from typing import TypeVar

from ..models.llama import LLaMACausalLM
from ..tokenizers.tokenizer import Tokenizer
from .default_generator import DefaultGenerator
from .hf_hub import FromHFHub

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="LLaMAGenerator")


class LLaMAGenerator(DefaultGenerator, FromHFHub):
    """
    Generator for LLaMa and Llama 2 model variants.
    """

    def __init__(self, tokenizer: Tokenizer, causal_lm: LLaMACausalLM):
        """
        Construct a LLaMA generator.

        :param tokenizer:
            A LLaMA tokenizer.
        :param causal_lm:
            A LLaMA causal language model.

        """
        super().__init__(
            tokenizer,
            causal_lm,
        )
