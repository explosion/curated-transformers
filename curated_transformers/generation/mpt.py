from typing import TypeVar

from ..models.mpt import MPTCausalLM
from ..tokenizers.tokenizer import Tokenizer
from .default_generator import DefaultGenerator
from .hf_hub import FromHF


class MPTGenerator(DefaultGenerator, FromHF):
    """
    Generator for MPT model variants.
    """

    def __init__(self, tokenizer: Tokenizer, causal_lm: MPTCausalLM):
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
