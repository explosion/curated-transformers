from typing import List
from abc import ABC, abstractmethod

from .config import GeneratorConfig


class GeneratorWrapper(ABC):
    """
    Model-specific wrapper for
    :class:`curated_transformers.generation.Generator`.
    """

    def __call__(self, prompts: List[str], config: GeneratorConfig) -> List[str]:
        """
        See the :meth:`.generate` method.
        """

        return self.generate(prompts, config)

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
