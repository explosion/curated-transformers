from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Set, cast

from torch.nn import Module

from .descriptor import SharedDataDescriptor, SharedDataType
from .logic import SharedModule, SharedParameter, SharingLogic


class Shareable(ABC):
    """
    Mixin class for models that share data between modules.

    A model using this mixin provides the descriptors of the
    shared data, and the mixin provides functions to share,
    unshare and clone the data.
    """

    _share_logic: List[SharingLogic]
    _active_metadata: List[Optional[Any]]
    _initialized: bool

    def __init__(self):
        self._share_logic = []
        self._active_metadata = []
        self._initialized = False

    @abstractmethod
    def shared_data(self) -> Set[SharedDataDescriptor]:
        """
        Return a set of shared data descriptors. This
        collection should be immutable, i.e., the output
        should not change after initialization.

        :returns:
            Set of shared data descriptors.
        """
        raise NotImplementedError

    def initialize_shared_data(self):
        """
        Prepare the model to perform data sharing.

        .. attention::
            This operation has to be performed *once* after the
            model's structure and datatype has been finalized.
        """
        if self._initialized:
            raise ValueError("Shareable data has already been initialized")

        shared_data = self.shared_data()
        self._check_for_overlaps(shared_data)
        self._share_logic = self._map_descriptors_to_logic(shared_data)
        for logic in self._share_logic:
            logic.initialize(model=cast(Module, self))
        self._initialized = True

    def tie_shared_data(self):
        """
        Shares data between modules.

        .. warning::
            Any operations that result in the modification of the model's
            submodules/parameters, e.g., moving to a different device, casting to
            a different datatype, etc, whilst tied could result in the shared data
            being cloned.
        """
        self._check_initalization()
        if self.tied:
            raise ValueError("Data is being actively shared - Unshare the data first")

        self._active_metadata = [None] * len(self._share_logic)
        for i, logic in enumerate(self._share_logic):
            metadata = logic.tie(model=cast(Module, self))
            self._active_metadata[i] = metadata

    def untie_shared_data(self):
        """
        Unshares data between modules.

        .. warning::
            Performing any computations using the untied model can lead to
            undefined behaviour. This method is primarily provided to prevent
            duplicate parameters being saved to disk during serialization.

        .. note::
            Once this operation is been performed, targets of shared parameters
            will be have zero-sized placeholder tensors. These can be removed from
            the state dict before serialization.
        """
        self._check_initalization()
        if not self.tied:
            raise ValueError("Data is not being actively shared - Share the data first")

        for logic, metadata in zip(self._share_logic, self._active_metadata):
            logic.untie(model=cast(Module, self), metadata=metadata)
        self._active_metadata.clear()

    def clone_shared_data(self):
        """
        Clone shared data into their targets. This is useful when
        exporting a model using ``torch.compile`` or TorchScript.

        .. attention::
            Once this operation has been performed, the model can no
            longer be tied or untied.
        """
        self._check_initalization()
        if self.tied:
            raise ValueError("Data is being actively shared - Unshare the data first")

        for logic in self._share_logic:
            logic.clone(model=cast(Module, self))

    @property
    def tied(self) -> bool:
        """
        Returns if the model is actively sharing data.
        """
        return len(self._active_metadata) > 0

    def _check_initalization(self):
        if not self._initialized:
            raise ValueError("Shareable data has not been initialized")

    @staticmethod
    def _map_descriptors_to_logic(
        descriptors: Iterable[SharedDataDescriptor],
    ) -> List[SharingLogic]:
        out: List[SharingLogic] = []
        for descriptor in descriptors:
            if descriptor.type == SharedDataType.MODULE:
                out.append(SharedModule(descriptor))
            elif descriptor.type == SharedDataType.PARAMETER:
                out.append(SharedParameter(descriptor))
            else:
                raise ValueError(f"Unexpected shared data type `{descriptor.type}`")
        return out

    @staticmethod
    def _check_for_overlaps(
        descriptors: Iterable[SharedDataDescriptor],
    ):
        modules = [d for d in descriptors if d.type == SharedDataType.MODULE]
        params = [d for d in descriptors if d.type == SharedDataType.PARAMETER]

        for param in params:
            for module in modules:
                if module.target in param.target:
                    raise ValueError(
                        f"Shared parameter '{param.target}' overlaps with "
                        f"shared module '{module.target}'"
                    )
