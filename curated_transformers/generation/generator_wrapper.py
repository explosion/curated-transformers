from typing import Iterator, List, Tuple
from abc import ABC, abstractmethod


class GeneratorWrapper(ABC):
    """
    Model-specific wrapper for
    :class:`curated_transformers.generation.Generator`.
    """

    def __call__(self, prompts: List[str]) -> List[str]:
        """
        See the :meth:`.generate` method.
        """

        return self.generate(prompts)

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """
        Generate text using the given prompts. This method returns the
        generated text for each prompt.

        :param prompts:
            Prompts to generate from.
        :returns:
            Strings generated for the prompts.
        """
        ...
