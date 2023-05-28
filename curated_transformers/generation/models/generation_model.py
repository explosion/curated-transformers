from typing import Iterator, List, Optional, Tuple, Type, TypeVar
from abc import ABC, abstractmethod
import torch


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="GenerationModel")


class GenerationModel(ABC):
    @classmethod
    @abstractmethod
    def from_hf_hub(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None
    ) -> Self:
        """Load a generation model from Huggingface Hub."""
        ...

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
