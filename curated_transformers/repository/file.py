import os
from abc import ABC, abstractmethod
from typing import IO, Optional


class RepositoryFile(ABC):
    """
    A repository file.

    Repository files can be a local path or a remote path exposed as a
    file-like object. This is a common base class for such different types
    of repository files.
    """

    @abstractmethod
    def open(self, mode: str = "rb", encoding: Optional[str] = None) -> IO:
        """
        Get the file as a file-like object.

        :param mode:
            Mode to open the file with (see Python ``open``).
        :param encoding:
            Encoding to use when the file is opened as text.
        :returns:
            An I/O stream.
        :raises FileNotFoundError:
            When the file cannot be found.
        :raises OSError:
            When the file cannot be opened.
        """
        ...

    @property
    @abstractmethod
    def path(self) -> Optional[str]:
        """
        Get the file as a local path.

        :returns:
            The repository file. If the file is not available as a local
            path, the value of this property is ``None``. In these cases
            ``open`` can be used to get the file as a file-like object.
        """
        ...

    @abstractmethod
    def exists(self) -> bool:
        """
        Returns if the file exists. This can cause the file
        to be cached locally.
        """
        ...


class LocalFile(RepositoryFile):
    """
    Repository file on the local machine.
    """

    def __init__(self, path: str):
        """
        Construct a local file representation.

        :param path:
            The path of the file on the local filesystem.
        """
        super().__init__()
        self._path = path

    def open(self, mode: str = "rb", encoding: Optional[str] = None) -> IO:
        # Raises OSError, so we don't have to do any rewrapping.
        return open(self._path, mode=mode, encoding=encoding)

    @property
    def path(self) -> Optional[str]:
        return self._path

    def exists(self) -> bool:
        return os.path.isfile(self._path)
