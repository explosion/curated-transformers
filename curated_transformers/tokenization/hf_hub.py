from typing import Any, Type, TypeVar
from abc import ABC, abstractmethod
from .._compat import has_hf_transformers, transformers


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FromPretrainedHFTokenizer")


class FromPretrainedHFTokenizer(ABC):
    """Mixin class for downloading tokenizers from Hugging Face Hub.

    A module using this mixin can implement the `convert_hf_tokenizer`
    method. The mixin will then provide the `from_hf_hub` method to
    download a tokenizer from the Hugging Face Hub.
    """

    @classmethod
    @abstractmethod
    def convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        """Create an instance of the tokenizer from the Hugging Face
        tokenizer instance.

        :param tokenizer:
            Hugging Face tokenizer.
        :return:
            New tokenizer.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_hub(cls: Type[Self], *, name: str, revision: str = "main") -> Self:
        """Construct a tokenizer and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :return:
            Module with the parameters loaded.
        """

        # We cannot directly use the Hugging Face Hub downloader like we do for the model
        # checkpoints as the tokenizer implementations do not use a consistent serialization
        # interface with stable filenames for artifacts.
        if not has_hf_transformers:
            raise ValueError(
                "`Loading tokenizers from Hugging Face Hub requires `transformers` package to be installed"
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        return cls.convert_hf_tokenizer(tokenizer)
