import json
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple

from .._compat import has_safetensors
from ..util.serde.checkpoint import ModelCheckpointType
from ._hf import (
    HF_MODEL_CONFIG,
    HF_TOKENIZER_CONFIG,
    PRIMARY_CHECKPOINT_FILENAMES,
    SHARDED_CHECKPOINT_INDEX_FILENAMES,
    SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY,
    SPECIAL_TOKENS_MAP,
    TOKENIZER_JSON,
)
from .file import RepositoryFile
from .transaction import TransactionContext


class Repository(ABC):
    """
    A repository that contains a model or tokenizer.
    """

    @abstractmethod
    def file(self, path: str) -> RepositoryFile:
        """
        Get a lazily-loaded repository file.

        :param path:
            The path of the file within the repository.
        :returns:
            The file.
        """
        ...

    def json_file(self, path: str) -> Dict[str, Any]:
        """
        Get and parse a JSON file.

        :param path:
            The path of the file within the repository.
        :returns:
            The deserialized JSON.
        :raises FileNotFoundError:
            When the file cannot be found.
        :raises OSError:
            When the file cannot be opened.
        :raises json.JSONDecodeError:
            When the JSON cannot be decoded.
        """
        file = self.file(path)
        with file.open("r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def pretty_path(self, path: Optional[str] = None) -> str:
        """
        Get a user-consumable path representation (e.g. for error messages).

        :param path:
            The path of a file within the repository. The repository path
            will be returned if ``path`` is falsy.
        :returns:
            The path representation.
        """
        ...

    @abstractmethod
    def transaction(self) -> TransactionContext:
        """
        Begins a new transaction. File operations performed on the transaction
        context will be deferred until the transaction completes successfully.

        :returns:
            The transaction context manager.
        """
        ...


class ModelRepository(Repository):
    """
    Repository wrapper that exposes some methods that are useful for working
    with repositories that contain a model.
    """

    # Cached model configuration.
    _model_config: Optional[Dict[str, Any]]

    def __init__(self, repo: Repository):
        """
        Construct a model repository wrapper.

        :param repo:
            The repository to wrap.
        """
        super().__init__()
        self.repo = repo
        self._model_config = None

    def file(self, filename: str) -> RepositoryFile:
        return self.repo.file(filename)

    def json_file(self, path: str) -> Dict[str, Any]:
        return self.repo.json_file(path)

    def model_checkpoints(self) -> Tuple[List[RepositoryFile], ModelCheckpointType]:
        """
        Retrieve the model checkpoints and checkpoint type.

        :returns:
            A tuple of the model checkpoints and the checkpoint type.
        :raises OSError:
            When the checkpoint paths files could not be retrieved.
        """

        def get_checkpoint_paths(
            checkpoint_type: ModelCheckpointType,
        ) -> List[RepositoryFile]:
            # Attempt to download a non-sharded checkpoint first.
            primary_ckpt = self.file(PRIMARY_CHECKPOINT_FILENAMES[checkpoint_type])
            if primary_ckpt.exists():
                return [primary_ckpt]

            # Try sharded checkpoint.
            index_filename = SHARDED_CHECKPOINT_INDEX_FILENAMES[checkpoint_type]

            try:
                index = self.json_file(index_filename)
            except (FileNotFoundError, JSONDecodeError, OSError) as e:
                raise OSError(
                    f"Index file for sharded checkpoint type {checkpoint_type.pretty_name} "
                    f"could not be loaded `{self.pretty_path(index_filename)}`"
                ) from e

            weight_map = index.get(SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY)
            if not isinstance(weight_map, dict):
                raise OSError(
                    f"Invalid index file in sharded {checkpoint_type.pretty_name} "
                    f"checkpoint for model at `{self.pretty_path()}`"
                )

            checkpoint_paths = []
            for filename in sorted(set(weight_map.values())):
                ckpt = self.file(filename)
                if not ckpt.exists():
                    raise OSError(
                        f"File for sharded checkpoint type {checkpoint_type.pretty_name} "
                        f"could not be found at `{self.pretty_path(index_filename)}`"
                    )
                checkpoint_paths.append(ckpt)

            return checkpoint_paths

        # Precedence: Safetensors > PyTorch
        checkpoint_type = ModelCheckpointType.SAFE_TENSORS
        checkpoint_paths: Optional[List[RepositoryFile]] = None
        if has_safetensors:
            try:
                checkpoint_paths = get_checkpoint_paths(checkpoint_type)
            except OSError:
                pass
        if checkpoint_paths is None:
            checkpoint_type = ModelCheckpointType.PYTORCH_STATE_DICT
            checkpoint_paths = get_checkpoint_paths(checkpoint_type)

        assert checkpoint_paths is not None
        assert checkpoint_type is not None
        return checkpoint_paths, checkpoint_type

    def model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration. The result is cached to speed up
        subsequent lookups.

        :returns:
            The model configuration.
        :raises OSError:
            When the model config cannot be opened.
        """
        if self._model_config is None:
            try:
                self._model_config = self.json_file(
                    path=HF_MODEL_CONFIG,
                )
            except (FileNotFoundError, JSONDecodeError, OSError) as e:
                raise OSError(
                    "Cannot load config for the model at "
                    f"`{self.pretty_path(HF_MODEL_CONFIG)}`"
                ) from e

        return self._model_config

    def model_type(self) -> str:
        """
        Get the model type.

        :returns:
            The model type.
        :raises OSError:
            When the model config cannot be opened.
        """
        return self.model_config()["model_type"]

    def pretty_path(self, path: Optional[str] = None) -> str:
        return self.repo.pretty_path(path)

    def transaction(self) -> TransactionContext:
        return self.repo.transaction()


class TokenizerRepository(Repository):
    """
    Repository wrapper that exposes some methods that are useful for working
    with repositories that contain a tokenizer.
    """

    _tokenizer_config: Optional[Dict[str, Any]]

    def __init__(self, repo: Repository):
        """
        Construct a tokenizer repository wrapper.

        :param repo:
            The repository to wrap.
        """
        super().__init__()
        self.repo = repo
        self._tokenizer_config = None

    def file(self, path: str) -> RepositoryFile:
        return self.repo.file(path)

    def json_file(self, path: str) -> Dict[str, Any]:
        return self.repo.json_file(path)

    def model_type(self) -> str:
        """
        Get the model type.

        :returns:
            The model type.
        :raises OSError:
            When the model config cannot be opened.
        """
        model_config = self.json_file(HF_MODEL_CONFIG)
        return model_config["model_type"]

    def pretty_path(self, path: Optional[str] = None) -> str:
        return self.repo.pretty_path(path)

    def special_tokens_map(self) -> Dict[str, Any]:
        """
        Return the tokenizer's special tokens map.

        :returns:
            The special tokens map.
        :raises OSError:
            When the special tokens map cannot be opened.
        """
        try:
            return self.repo.json_file(
                path=SPECIAL_TOKENS_MAP,
            )
        except (FileNotFoundError, JSONDecodeError, OSError) as e:
            raise OSError(
                "Could not load special tokens map for the tokenizer at "
                f"`{self.repo.pretty_path(SPECIAL_TOKENS_MAP)}`"
            ) from e

    def tokenizer_config(self) -> Dict[str, Any]:
        """
        Return the model's tokenizer configuration. The result is cached to
        speed up subsequent lookups.

        :returns:
            Model configuration.
        :raises OSError:
            When the tokenizer config cannot be opened.
        """
        if self._tokenizer_config is None:
            try:
                self._tokenizer_config = self.repo.json_file(
                    path=HF_TOKENIZER_CONFIG,
                )
            except (FileNotFoundError, JSONDecodeError, OSError) as e:
                raise OSError(
                    "Couldn't find a valid config for the tokenizer at "
                    f"`{self.repo.pretty_path(HF_TOKENIZER_CONFIG)}`"
                ) from e

        return self._tokenizer_config

    def tokenizer_json(self) -> RepositoryFile:
        """
        Return the HF tokenizers' ``tokenizer.json``.

        :returns:
            The tokenizer file.
        """
        return self.repo.file(TOKENIZER_JSON)

    def transaction(self) -> TransactionContext:
        return self.repo.transaction()
