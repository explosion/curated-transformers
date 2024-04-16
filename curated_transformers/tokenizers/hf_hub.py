from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from fsspec import AbstractFileSystem
from huggingface_hub.utils import EntryNotFoundError

from ..repository.file import RepositoryFile
from ..repository.fsspec import FsspecArgs, FsspecRepository
from ..repository.hf_hub import HfHubRepository
from ..repository.repository import Repository, TokenizerRepository

SelfFromHF = TypeVar("SelfFromHF", bound="FromHF")


class FromHF(ABC):
    """
    Mixin class for downloading tokenizers from Hugging Face Hub.

    It directly queries the Hugging Face Hub to load the tokenizer from
    its configuration file.
    """

    @classmethod
    @abstractmethod
    def from_hf_hub_to_cache(
        cls: Type[SelfFromHF],
        *,
        name: str,
        revision: str = "main",
    ):
        """
        Download the tokenizer's serialized model, configuration and vocab files
        from Hugging Face Hub into the local Hugging Face cache directory.
        Subsequent loading of the tokenizer will read the files from disk. If the
        files are already cached, this is a no-op.

        :param name:
            Model name.
        :param revision:
            Model revision.
        """
        raise NotImplementedError

    @classmethod
    def from_fsspec(
        cls: Type[SelfFromHF],
        *,
        fs: AbstractFileSystem,
        model_path: str,
        fsspec_args: Optional[FsspecArgs] = None,
    ) -> SelfFromHF:
        """
        Construct a tokenizer and load its parameters from an fsspec filesystem.

        :param fs:
            Filesystem.
        :param model_path:
            The model path.
        :param fsspec_args:
            Implementation-specific keyword arguments to pass to fsspec
            filesystem operations.
        :returns:
            The tokenizer.
        """
        return cls.from_repo(
            repo=FsspecRepository(fs, model_path, fsspec_args),
        )

    @classmethod
    def from_hf_hub(
        cls: Type[SelfFromHF], *, name: str, revision: str = "main"
    ) -> SelfFromHF:
        """
        Construct a tokenizer and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :returns:
            The tokenizer.
        """
        return cls.from_repo(
            repo=HfHubRepository(name=name, revision=revision),
        )

    @classmethod
    @abstractmethod
    def from_repo(
        cls: Type[SelfFromHF],
        repo: Repository,
    ) -> SelfFromHF:
        """
        Construct and load a tokenizer from a repository.

        :param repository:
            The repository to load from.
        :returns:
            Loaded tokenizer.
        """
        ...


SelfLegacyFromHF = TypeVar("SelfLegacyFromHF", bound="LegacyFromHF")


class LegacyFromHF(FromHF):
    """
    Subclass of :class:`.FromHF` for legacy tokenizers. This subclass
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
        cls: Type[SelfLegacyFromHF],
        *,
        vocab_files: Mapping[str, RepositoryFile],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> SelfLegacyFromHF:
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
    def from_hf_hub_to_cache(
        cls: Type[SelfLegacyFromHF],
        *,
        name: str,
        revision: str = "main",
    ):
        repo = TokenizerRepository(HfHubRepository(name, revision=revision))
        for _, filename in cls.vocab_files.items():
            _ = repo.file(filename)

        try:
            _ = repo.tokenizer_config()
        except EntryNotFoundError:
            pass

    @classmethod
    def from_repo(
        cls: Type[SelfLegacyFromHF],
        repo: Repository,
    ) -> SelfLegacyFromHF:
        repo = TokenizerRepository(repo)
        vocab_files = {}
        for vocab_file, filename in cls.vocab_files.items():
            vocab_files[vocab_file] = repo.file(filename)

        try:
            tokenizer_config = repo.tokenizer_config()
        except OSError:
            tokenizer_config = None

        return cls._load_from_vocab_files(
            vocab_files=vocab_files, tokenizer_config=tokenizer_config
        )
