from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Set, Type

from torch.nn import Identity, Module

from ..util.pytorch import ModuleIterator, apply_to_module


class WithSharedParameters(ABC):
    """
    Mixin class for models that share parameters between modules.

    A modle using this mixin provides the necessary information
    to share parameters between different submodules. submodules
    that use the parameters of others are initialized with the
    :py:class:`~curated_transformers.models.SharedModulePlaceholder`
    class.
    """

    _tied_modules: Set[str]

    def __init__(self):
        self._tied_modules = {}

    @staticmethod
    def _validate_source_and_target_for_typing_cloning(
        source: Module, target: Module, source_prefix: str, target_prefix: str
    ):
        if not isinstance(source, SharedModulePlaceholder):
            raise ValueError(
                f"Shared module '{source_prefix}' expected to be of type "
                f"`{SharedModulePlaceholder.__name__}`, but got `{type(source).__name__}`"
            )

        shared_module: SharedModulePlaceholder = source
        if type(target) != source.target_type:
            raise ValueError(
                f"Shared module '{source_prefix}' expected a target of type "
                f"`{shared_module.target_type.__name__}`, but got `{type(target).__name__}`"
            )

        if source_prefix in target_prefix:
            raise ValueError(
                f"Target of shared module ('{target_prefix}') cannot contain the "
                f"shared module itself ('{source_prefix}')"
            )

    def _get_target_module(self, source_prefix: str, target_prefix: str) -> Module:
        try:
            self_module: Module = self  # type: ignore[assignment]
            return self_module.get_submodule(target_prefix)
        except AttributeError:
            raise ValueError(
                f"Couldn't resolve target module '{target_prefix}' for "
                f"shared module '{source_prefix}'"
            )

    @abstractmethod
    def shared_module_targets(self) -> Dict[str, str]:
        """
        Return a dictionary mapping the prefixes of
        :py:class:`~curated_transformers.models.SharedModulePlaceholder`
        modules to the prefixes of their target module.

        :returns:
            Dictionary of shared parameter targets.
        """
        raise NotImplementedError

    def tie_parameters(self):
        """
        Tie the parameters of the shared submodules.
        """
        targets = self.shared_parameter_targets()
        if len(targets) == 0:
            raise ValueError(f"No shareable parameters found")
        elif len(self._tied_modules) > 0:
            raise ValueError("Shared parameters have already been tied together")

        def apply(itr: ModuleIterator):
            if itr.parent is None:
                return
            elif itr.prefix not in targets:
                return

            target_prefix = targets[itr.prefix]
            target_module = self._get_target_module(
                source_prefix=itr.prefix, target_prefix=target_prefix
            )

            self._validate_source_and_target_for_typing_cloning(
                source=itr.module,
                target=target_module,
                source_prefix=itr.prefix,
                target_prefix=target_prefix,
            )

            # TODO: Will need special handling when we implement fine-tuning.
            itr.parent._modules[itr.name] = target_module

        apply_to_module(self, apply)

    def untie_parameters(self):
        """
        Untie the tied parameters of the shared submodules.
        This is useful when serializing the model to disk.
        """
        if len(self._tied_modules) == 0:
            raise ValueError("Sharable parameters need to be tied first")

        def apply(itr: ModuleIterator):
            if itr.parent is None:
                return
            elif itr.prefix not in self._tied_modules:
                return

            target_module_cls = type(itr.module)
            itr.parent._modules[itr.name] = SharedModulePlaceholder(target_module_cls)

        apply_to_module(self, apply)

    def clone_parameters(self):
        """
        Perform a deep-copy of the shared submodules into their corresponding
        placeholders. This is useful when exporting a model using ``torch.compile``
        or TorchScript.
        """
        targets = self.shared_parameter_targets()
        if len(targets) == 0:
            raise ValueError(f"No shareable parameters found")
        elif len(self._tied_modules) > 0:
            raise ValueError("Shared parameters have already been tied together")

        def apply(itr: ModuleIterator):
            if itr.parent is None:
                return
            elif itr.prefix not in targets:
                return

            target_prefix = targets[itr.prefix]
            target_module = self._get_target_module(
                source_prefix=itr.prefix, target_prefix=target_prefix
            )

            self._validate_source_and_target_for_typing_cloning(
                source=itr.module,
                target=target_module,
                source_prefix=itr.prefix,
                target_prefix=target_prefix,
            )

            itr.parent._modules[itr.name] = deepcopy(target_module)

        apply_to_module(self, apply)


class SharedModulePlaceholder(Identity):
    """
    Placeholder for modules that use a different module's parameters.
    """

    def __init__(self, target_module_cls: Type[Module], *args, **kwargs):
        super().__init__(*args, **kwargs)

        if issubclass(target_module_cls, SharedModulePlaceholder):
            raise ValueError(
                "Share module placeholder cannot target another placeholder"
            )

        self._target_module_cls = target_module_cls

    @property
    def target_type(self) -> Type[Module]:
        return self._target_module_cls
