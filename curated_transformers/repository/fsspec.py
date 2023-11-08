import warnings
from dataclasses import dataclass
from typing import IO, Any, Dict, Optional

from fsspec import AbstractFileSystem

from .repository import Repository, RepositoryFile
from .transaction import TransactionContext


@dataclass
class FsspecArgs:
    """
    Convenience wrapper for additional fsspec arguments.
    """

    kwargs: Dict[str, Any]

    def __init__(self, **kwargs):
        """
        Keyword arguments are passed through to the fsspec implementation.
        Construction may raise in the future when reserved arguments like
        ``mode`` or ``encoding`` are used.
        """
        # Future improvement: raise on args that are used by fsspec methods,
        # e.g. `mode` or `encoding`.
        self.kwargs = kwargs


class FsspecFile(RepositoryFile):
    """
    Repository file on an `fsspec`_ filesystem.

    .. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/
    """

    def __init__(
        self,
        fs: AbstractFileSystem,
        path: str,
        fsspec_args: Optional[FsspecArgs] = None,
    ):
        """
        Construct an fsspec file representation.

        :param fs:
            The filesystem.
        :param path:
            The path of the file on the filesystem.
        :param fsspec_args:
            Implementation-specific arguments to pass to fsspec filesystem
            operations.
        """
        super().__init__()
        self._fs = fs
        self._path = path
        self._fsspec_args = FsspecArgs() if fsspec_args is None else fsspec_args

    def open(self, mode: str = "rb", encoding: Optional[str] = None) -> IO:
        if ("r" in mode or "+" in mode) and not self.exists():
            raise FileNotFoundError(
                f"Cannot find fsspec path {self._fs.unstrip_protocol(self._path)}"
            )

        try:
            return self._fs.open(
                self._path, mode=mode, encoding=encoding, **self._fsspec_args.kwargs
            )
        except Exception as e:
            raise OSError(
                f"Cannot open fsspec path {self._fs.unstrip_protocol(self._path)}"
            ) from e

    @property
    def path(self) -> Optional[str]:
        return None

    def exists(self) -> bool:
        return self._fs.exists(self._path, **self._fsspec_args.kwargs)


class FsspecRepository(Repository):
    """
    Repository using a filesystem that uses the `fsspec`_ interface.

    .. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/

    :param fs:
        The filesystem.
    :param path:
        The the path of the repository within the filesystem.
    :param fsspec_args:
        Additional arguments that should be passed to the fsspec
        implementation.
    """

    def __init__(
        self,
        fs: AbstractFileSystem,
        path: str,
        fsspec_args: Optional[FsspecArgs] = None,
    ):
        super().__init__()
        self.fs = fs
        self.repo_path = path
        self.fsspec_args = FsspecArgs() if fsspec_args is None else fsspec_args

    def file(self, path: str) -> RepositoryFile:
        full_path = f"{self.repo_path}/{path}"
        return FsspecFile(self.fs, full_path, self.fsspec_args)

    def pretty_path(self, path: Optional[str] = None) -> str:
        if not path:
            return self._protocol
        return f"{self._protocol}/{path}"

    def transaction(self) -> TransactionContext:
        return FsspecTransactionContext(self)

    @property
    def _protocol(self) -> str:
        return self.fs.unstrip_protocol(self.repo_path)


class FsspecTransactionContext(TransactionContext):
    """
    `fsspec`_ transaction context manager.
    TODO: Implement/currently a noop

    .. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/
    """

    def __init__(self, repo: FsspecRepository):
        """
        :param repo:
            The parent repository.
        """
        super().__init__()
        self._repo = repo

    def open(self, path: str, mode: str, encoding: Optional[str] = None) -> IO:
        return self._repo.file(path).open(mode=mode, encoding=encoding)

    @property
    def repo(self) -> Repository:
        return self._repo

    def __enter__(self):
        warnings.warn("Transaction support is currently not implemented for fsspec")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass
