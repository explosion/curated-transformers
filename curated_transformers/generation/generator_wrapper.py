from typing import Iterator, List, Tuple
from abc import ABC, abstractmethod


class GeneratorWrapper(ABC):
    """
    Model-specific wrapper for
    :class:`curated_transformers.generation.Generator`.
    """

    def __call__(self, prompts: List[str]) -> Iterator[List[Tuple[int, str]]]:
        """
        See the :meth:`.generate` method.
        """

        return self.generate(prompts)

    @abstractmethod
    def generate(self, prompts: List[str]) -> Iterator[List[Tuple[int, str]]]:
        """
        Generate text using the given prompts. This function yields for
        each generation step a list of requence identifiers and the
        corresponding generated substring.

        :param prompts:
            Prompts to generate from.
        :returns:
            An iterator returning for each generation step sequence
            identifiers and the substrings that were generated
            for the sequences.
        """
        ...
