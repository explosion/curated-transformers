from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from huggingface_hub.utils import EntryNotFoundError

from ..util.hf import get_tokenizer_config, hf_hub_download

SelfFromHFHub = TypeVar("SelfFromHFHub", bound="FromHFHub")


class FromHFHub(ABC):
    """
    Mixin class for downloading tokenizers from Hugging Face Hub.

    It directly queries the Hugging Face Hub to load the tokenizer from
    its configuration file.
    """

    @classmethod
    @abstractmethod
    def from_hf_hub(
        cls: Type[SelfFromHFHub], *, name: str, revision: str = "main"
    ) -> SelfFromHFHub:
        """
        Construct a tokenizer and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :returns:
            The tokenizer.
        """
        raise NotImplementedError


SelfLegacyFromHFHub = TypeVar("SelfLegacyFromHFHub", bound="LegacyFromHFHub")


class LegacyFromHFHub(FromHFHub):
    """
    Subclass of :class:`.FromHFHub` for legacy tokenizers. This subclass
    implements the ``from_hf_hub`` method and provides through the abstract
    ``_load_from_vocab_files`` method:

    * The vocabulary files requested by a tokenizer through the
    ``vocab_files`` member variable.
    * The tokenizer configuration (when available).
    """

    vocab_files: Dict[str, str] = {}

    @classmethod
    @abstractmethod
    def _load_from_vocab_files(
        cls: Type[SelfLegacyFromHFHub],
        *,
        vocab_files: Dict[str, Path],
        tokenizer_config: Optional[Dict[str, Any]]
    ) -> SelfLegacyFromHFHub:
        """
        Construct a tokenizer from its vocabulary files and optional
        configuration.

        :param vocab_files:
            The resolved vocabulary files (in a local cache).
        :param tokenizer_config:
            The tokenizer configuration (when available).
        :returns:
            The tokenizer.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_hub(
        cls: Type[SelfLegacyFromHFHub], *, name: str, revision: str = "main"
    ) -> SelfLegacyFromHFHub:
        """
        Construct a tokenizer and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :returns:
            The tokenizer.
        """
        vocab_files = {}
        for vocab_file, filename in cls.vocab_files.items():
            vocab_files[vocab_file] = Path(
                hf_hub_download(repo_id=name, filename=filename, revision=revision)
            )

        # Try to get the tokenizer configuration.
        try:
            tokenizer_config = get_tokenizer_config(name=name, revision=revision)
        except EntryNotFoundError:
            tokenizer_config = None

        return cls._load_from_vocab_files(
            vocab_files=vocab_files, tokenizer_config=tokenizer_config
        )
