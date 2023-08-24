from copy import deepcopy
from typing import Any, List, Optional, Type

from torch.nn import Identity, Module

from ...quantization import is_quantized_module
from ...util.pytorch import split_full_name
from ..descriptor import SharedDataDescriptor, SharedDataType
from .abc import SharingLogic


class SharedModulePlaceholder(Identity):
    """
    Placeholder for modules that are to be replaced with another module.
    """

    _wrapped_module: List[Module]

    def __init__(self, wrapped_module: Module, *args, **kwargs):
        """
        Construct a shared module placeholder.

        :param wrapped_module:
            The original module in the model that will be
            replaced.
        :returns:
            The placeholder
        """
        super().__init__(*args, **kwargs)

        if isinstance(wrapped_module, SharedModulePlaceholder):
            raise ValueError(
                "Shared module placeholder cannot target another placeholder"
            )
        elif any(not x.is_meta for x in wrapped_module.parameters()):
            # This ensures that the placeholder initialization happens
            # before the model is initialized using its checkpoint(s).
            raise ValueError(
                "Attempting to replace a non-meta shared module with a placeholder"
            )

        # Needed to work around the automatic registration of `Module` attributes.
        self._wrapped_module = [wrapped_module]

    @property
    def wrapped_source_type(self) -> Type[Module]:
        return type(self._wrapped_module[0])


class SharedModule(SharingLogic):
    def __init__(
        self,
        descriptor: SharedDataDescriptor,
    ):
        assert descriptor.type == SharedDataType.MODULE
        self.source = descriptor.source
        self.target = descriptor.target

        invalid_name_msg = (
            "Invalid shared module name '{name}' - Must contain at least one dot"
        )

        if not self.target or len(self.target.split(".")) < 2:
            raise ValueError(invalid_name_msg.format(name=self.target))
        elif not self.source or len(self.source.split(".")) < 2:
            raise ValueError(invalid_name_msg.format(name=self.source))
        elif self.target in self.source:
            raise ValueError(
                f"Source of a shared module ('{self.source}') cannot contain the "
                f"shared module itself ('{self.target}')"
            )

    def _get_source_module(self, root: Module) -> Module:
        try:
            return root.get_submodule(self.source)
        except AttributeError:
            raise ValueError(
                f"Couldn't resolve source module '{self.source}' for "
                f"shared module '{self.target}'"
            )

    def _get_target_module(self, root: Module) -> Module:
        try:
            return root.get_submodule(self.target)
        except AttributeError:
            raise ValueError(
                f"Couldn't resolve shared module '{self.target}' for "
                f"source module '{self.source}'"
            )

    def _replace_target_module(self, model: Module, replacement: Module):
        target_name, parent_name = split_full_name(self.target)
        parent_module = model.get_submodule(parent_name)
        parent_module._modules[target_name] = replacement

    def _validate_shareability(self, module: Module):
        if is_quantized_module(module):
            raise ValueError(
                f"Cannot share data from/with a quantized module of type `{type(module).__name__}`"
            )

    def _tie_or_clone(
        self,
        model: Module,
        clone: bool,
    ) -> Optional[Any]:
        source_module = self._get_source_module(model)
        target_module = self._get_target_module(model)

        if not isinstance(target_module, SharedModulePlaceholder):
            raise ValueError(
                f"Shared module '{self.target}' expected to be of type "
                f"`{SharedModulePlaceholder.__name__}`, but got `{type(target_module).__name__}`"
            )
        assert isinstance(target_module, SharedModulePlaceholder)
        if type(source_module) != target_module.wrapped_source_type:
            raise ValueError(
                f"Shared module '{self.target}' expected a source of type "
                f"`{target_module.wrapped_source_type.__name__}`, but got `{type(source_module).__name__}`"
            )
        self._validate_shareability(source_module)

        if clone:
            self._replace_target_module(model, deepcopy(source_module))
            return None
        else:
            # TODO: Tying will probably need special handling to support fine-tuning.
            self._replace_target_module(model, source_module)
            return target_module

    def initialize(self, model: Module):
        target_module = self._get_target_module(model)
        self._validate_shareability(target_module)

        placeholder = SharedModulePlaceholder(target_module)
        self._replace_target_module(model, placeholder)

    def tie(self, model: Module) -> Optional[Any]:
        return self._tie_or_clone(model, clone=False)

    def untie(
        self,
        model: Module,
        metadata: Optional[Any] = None,
    ):
        target_module = self._get_target_module(model)
        if isinstance(target_module, SharedModulePlaceholder):
            raise ValueError(
                f"Shared module '{self.target}' was not tied before untying"
            )

        assert isinstance(metadata, SharedModulePlaceholder)
        self._replace_target_module(model, metadata)

    def clone(self, model: Module):
        self._tie_or_clone(model, clone=True)
