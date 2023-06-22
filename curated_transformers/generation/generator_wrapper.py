from abc import ABC, abstractmethod
from typing import List

from torch.nn import Module

from curated_transformers.generation.string_generator import StringGenerator

from .config import GeneratorConfig


class GeneratorWrapper(ABC):
    """
    Model-specific wrapper for
    :class:`curated_transformers.generation.Generator`.
    """

    generator: StringGenerator

    def __init__(self, generator: StringGenerator):
        self.generator = generator

    def __call__(self, prompts: List[str], config: GeneratorConfig) -> List[str]:
        """
        See the :meth:`.generate` method.
        """

        return self.generate(prompts, config)

    def eval(self):
        """Set the wrapped model to evaluation mode."""
        self.generator.eval()

    @abstractmethod
    def generate(self, prompts: List[str], config: GeneratorConfig) -> List[str]:
        """
        Generate text using the given prompts. This method returns the
        generated text for each prompt.

        :param prompts:
            Prompts to generate from.
        :param config:
            Generator configuraton.
        :returns:
            Strings generated for the prompts.
        """
        ...

    def train(self, mode: bool = True):
        """
        Set the wrapped model's train mode.

        :param mode:
            Set to training mode when ``True`` or evaluation otherwise.
        """
        self.generator.train(mode)
