from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar

import torch

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FromHFHub")


class FromHFHub(ABC):
    @classmethod
    @abstractmethod
    def from_hf_hub(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None
    ) -> Self:
        """Load a generator from Huggingface Hub."""
        ...
