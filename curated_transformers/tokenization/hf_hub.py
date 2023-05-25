from typing import Any, Dict, Type, TypeVar
from abc import ABC, abstractmethod
from huggingface_hub import hf_hub_download
import json
from pathlib import Path

from .._compat import has_hf_transformers, transformers

HF_TOKENIZER_FILENAME = "tokenizer.json"

# Only provided as typing.Self in Python 3.11+.
SelfFromPretrainedTokenizer = TypeVar(
    "SelfFromPretrainedTokenizer", bound="FromPretrainedHFTokenizer"
)
SelfFromHFHub = TypeVar("SelfFromHFHub", bound="FromHFHub")


class FromHFHub(ABC):
    """
    Mixin class for downloading tokenizers from Hugging Face Hub.
    """

    @classmethod
    def from_tokenizer_json_file(
        cls: Type[SelfFromHFHub],
        path: Path,
    ) -> SelfFromHFHub:
        """Construct a Hugging Face fast tokenizer JSON file.

        :param tokenizer_path:
            Path to the tokenizer JSON file.
        :returns:
            The tokenizer.
        """
        with open(path, "r") as f:
            hf_tokenizer = json.load(f)
        if not isinstance(hf_tokenizer, dict):
            raise ValueError(
                f"Tokenizer must be a json dict, was: {type(hf_tokenizer).__name__}"
            )

        return cls._convert_hf_tokenizer_json(hf_tokenizer=hf_tokenizer)

    @classmethod
    @abstractmethod
    def _convert_hf_tokenizer_json(
        cls: Type[SelfFromHFHub], *, hf_tokenizer: Dict[str, Any]
    ) -> SelfFromHFHub:
        """
        Construct a tokenizer from a deserialized Hugging Face fast tokenizer
        deserialized JSON file.

        :param hf_tokenizer:
            Deserialized tokenizer JSON.
        :returns:
            The tokenizer.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_hub(
        cls: Type[SelfFromHFHub], *, name: str, revision: str = "main"
    ) -> SelfFromHFHub:
        """Construct a tokenizer and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :return:
            The tokenizer.
        """
        tokenizer_filename = _get_tokenizer_filepath(name, revision)
        with open(tokenizer_filename, "r") as f:
            hf_tokenizer = json.load(f)
        if not isinstance(hf_tokenizer, dict):
            raise ValueError(
                f"Tokenizer must be a json dict, was: {type(hf_tokenizer).__name__}"
            )
        return cls._convert_hf_tokenizer_json(hf_tokenizer=hf_tokenizer)


def _get_tokenizer_filepath(name: str, revision: str) -> str:
    try:
        return hf_hub_download(
            repo_id=name, filename=HF_TOKENIZER_FILENAME, revision=revision
        )
    except:
        raise ValueError(
            f"Couldn't find a valid config for model `{name}` "
            f"(revision `{revision}`) on HuggingFace Model Hub"
        )


class FromPretrainedHFTokenizer(ABC):
    """Mixin class for downloading tokenizers from Hugging Face Hub.

    A module using this mixin can implement the `convert_hf_tokenizer`
    method. The mixin will then provide the `from_hf_hub` method to
    download a tokenizer from the Hugging Face Hub.
    """

    @classmethod
    @abstractmethod
    def _convert_hf_tokenizer(
        cls: Type[SelfFromPretrainedTokenizer], tokenizer: Any
    ) -> SelfFromPretrainedTokenizer:
        """Create an instance of the tokenizer from the Hugging Face
        tokenizer instance.

        :param tokenizer:
            Hugging Face tokenizer.
        :return:
            New tokenizer.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_tokenizer(
        cls: Type[SelfFromPretrainedTokenizer], *, name: str, revision: str = "main"
    ) -> SelfFromPretrainedTokenizer:
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
        return cls._convert_hf_tokenizer(tokenizer)
