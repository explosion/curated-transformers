from abc import ABC, abstractmethod
from typing import Any, Optional

from torch.nn import Module


class SharingLogic(ABC):
    """
    Base class that defines sharing logic.
    """

    @abstractmethod
    def initialize(self, model: Module):
        """
        Perform initialization logic. Called during model
        initialization, after the model's structure has been
        finalized.

        :param model:
            Top-level module that represents the full model.
        """
        raise NotImplementedError

    @abstractmethod
    def tie(self, model: Module) -> Optional[Any]:
        """
        Determine if data is to be shared with a given module
        and execute the sharing logic.

        :param model:
            Top-level module that represents the full model.
        :returns:
            Metadata to be used when untying.
        """
        raise NotImplementedError

    @abstractmethod
    def untie(
        self,
        model: Module,
        metadata: Optional[Any] = None,
    ):
        """
        Determine if data has already been shared with a given module
        and execute the unsharing logic. This is useful when serializing
        the model to disk.

        :param model:
            Top-level module that represents the full model.
        :param metadata:
            Metadata returned by the tying process.
        """
        raise NotImplementedError

    @abstractmethod
    def clone(self, model: Module):
        """
        Determine if data is to be shared with a given module
        and execute the cloning logic. This is useful when exporting
        a model using ``torch.compile`` or TorchScript.

        :param model:
            Top-level module that represents the full model.
        """
        raise NotImplementedError
