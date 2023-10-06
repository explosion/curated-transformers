from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .repository import Repository


class TransactionContext(ABC):
    """
    A context manager that represents an active transaction in
    a repository.
    """

    @abstractmethod
    def open(self, path: str, mode: str, encoding: Optional[str] = None) -> IO:
        """
        Opens a file as a part of a transaction. Changes to the
        file are deferred until the transaction has completed
        successfully.

        :param path:
            The path to the file on the parent repository.
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
    def repo(self) -> "Repository":
        """
        :returns:
            The parent repository on which this transaction is performed.
        """
        ...

    @abstractmethod
    def __enter__(self):
        """
        Invoked when the context manager is entered.
        """
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Invoked when the context manager exits.
        """
        ...
