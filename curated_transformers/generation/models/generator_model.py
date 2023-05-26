from typing import Iterator, List, Tuple, Type, TypeVar
from abc import ABC, abstractmethod
import torch


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="GeneratorModel")


class GeneratorModel(ABC):
    @classmethod
    @abstractmethod
    def from_hf_hub(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
        device: torch.device = torch.device("cuda")
    ) -> Self:
        ...

    def __call__(self, prompts: List[str]) -> Iterator[Iterator[Tuple[int, str]]]:
        return self.generate(prompts)

    @abstractmethod
    def generate(self, prompts: List[str]) -> Iterator[Iterator[Tuple[int, str]]]:
        ...
