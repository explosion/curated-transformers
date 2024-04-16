from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar

import torch

from ..quantization.bnb.config import BitsAndBytesConfig

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FromHF")


class FromHF(ABC):
    """
    Mixin class for downloading generators from Hugging Face Hub.

    It automatically infers the tokenizer and model type and loads
    the parameters and the configuration from the hub.
    """

    @classmethod
    @abstractmethod
    def from_hf_hub_to_cache(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
    ):
        """
        Download the generator's model and tokenizer from Hugging Face Hub
        into the local Hugging Face cache directory. Subsequent loading of the
        generator will load the model and the tokenizer from disk.

        :param name:
            Model name.
        :param revision:
            Model revision.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hf_hub(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
        device: Optional[torch.device] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
    ) -> Self:
        """
        Construct a generator and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :param device:
            Device on which to initialize the model.
        :param quantization_config:
            Configuration for loading quantized weights.
        :returns:
            Generator with the parameters loaded.
        """
        raise NotImplementedError
