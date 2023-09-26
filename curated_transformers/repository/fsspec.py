from dataclasses import dataclass
from typing import IO, Any, Dict, Optional

from fsspec import AbstractFileSystem

from .repository import Repository, RepositoryFile


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
        try:
            return self._fs.open(
                self._path, mode=mode, encoding=encoding, **self._fsspec_args.kwargs
            )
        except Exception as e:
            raise OSError(
                f"Cannot open fsspec path {self._fs.unstrip_protocol(self.path)}"
            )

    @property
    def path(self) -> Optional[str]:
        return None


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
        if not self.fs.exists(full_path, **self.fsspec_args.kwargs):
            raise FileNotFoundError(f"Cannot find file in repository: {path}")
        return FsspecFile(self.fs, full_path, self.fsspec_args)

    def pretty_path(self, path: Optional[str] = None) -> str:
        if not path:
            return self._protocol
        return f"{self._protocol}/{path}"

    @property
    def _protocol(self) -> str:
        return self.fs.unstrip_protocol(self.repo_path)
