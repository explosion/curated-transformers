from abc import ABC, abstractmethod
from typing import List

from .config import GeneratorConfig


class GeneratorWrapper(ABC):
    """
    Model-specific wrapper for :class:`curated_transformers.generation.Generator`.
    """

    def __call__(self, prompts: List[str], config: GeneratorConfig) -> List[str]:
        """
        Alias for :meth:`.generate`.
        """

        return self.generate(prompts, config)

    @abstractmethod
    def generate(self, prompts: List[str], config: GeneratorConfig) -> List[str]:
        """
        Generate text using the given prompts.

        :param prompts:
            Prompts to generate from.
        :param config:
            Generator configuraton.
        :returns:
            Strings generated for the prompts.
        """
        ...
