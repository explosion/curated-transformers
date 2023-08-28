from dataclasses import dataclass
from typing import Any, Optional, Type

import torch
from torch.nn import Module, Parameter

from ...quantization import is_quantized_module, is_quantized_parameter
from ...util.pytorch import split_full_name
from ..descriptor import SharedDataDescriptor, SharedDataType
from .abc import SharingLogic


class SharedParameterPlaceholder(Parameter):
    """
    Placeholder for parameters that are to be replaced with another parameter.
    """

    _wrapped_param: Parameter

    @classmethod
    def for_param(cls, wrapped_param: Parameter) -> "SharedParameterPlaceholder":
        """
        Construct a shared parameter placeholder.

        :param wrapped_param:
            The original parameter in the model that will be
            replaced.
        :returns:
            The placeholder
        """
        # We can't use a regular init method due to how Parameter
        # instantiates it calls, which results in the wrapped
        # parameter getting cloned into the placeholder's backing store/

        if isinstance(wrapped_param, SharedParameterPlaceholder):
            raise ValueError(
                "Shared parameter placeholder cannot target another placeholder"
            )
        elif not wrapped_param.is_meta:
            # This ensures that the placeholder initialization happens
            # before the model is initialized using its checkpoint(s).
            raise ValueError(
                "Attempting to replace a non-meta shared parameter with a placeholder"
            )

        instance = cls()
        instance._wrapped_param = wrapped_param
        return instance

    @property
    def wrapped_shape(self) -> torch.Size:
        return self._wrapped_param.shape

    @property
    def wrapped_dtype(self) -> torch.dtype:
        return self._wrapped_param.dtype


class SharedParameter(SharingLogic):
    def __init__(
        self,
        descriptor: SharedDataDescriptor,
    ):
        assert descriptor.type == SharedDataType.PARAMETER
        self.source = descriptor.source
        self.target = descriptor.target

        invalid_name_msg = (
            "Invalid shared parameter name '{name}' - Must contain at least one dot"
        )

        if len(self.source.split(".")) < 2:
            raise ValueError(invalid_name_msg.format(name=self.source))
        elif len(self.target.split(".")) < 2:
            raise ValueError(invalid_name_msg.format(name=self.target))
        elif self.target == self.source:
            raise ValueError(
                f"Source parameter name ('{self.source}') cannot "
                "be the same as the target parameter name"
            )

    def _get_source_parameter(self, root: Module) -> Parameter:
        try:
            return root.get_parameter(self.source)
        except AttributeError:
            raise ValueError(
                f"Couldn't resolve source parameter '{self.source}' for "
                f"target parameter '{self.target}'"
            )

    def _get_target_parameter(self, root: Module) -> Parameter:
        try:
            return root.get_parameter(self.target)
        except AttributeError:
            raise ValueError(
                f"Couldn't resolve target parameter '{self.target}' for "
                f"source parameter '{self.source}'"
            )

    def _validate_shareability(self, param: Parameter):
        if is_quantized_parameter(param):
            raise ValueError(
                f"Cannot share data from/with a quantized parameter of type `{type(param).__name__}`"
            )
        elif type(param) != Parameter:
            raise ValueError(
                f"Shared parameter must be of type '{Parameter.__name__}', got '{type(param).__name__}'"
            )

    def _replace_target_parameter(self, model: Module, replacement: Parameter):
        target_name, parent_name = split_full_name(self.target)
        parent_module = model.get_submodule(parent_name)
        parent_module._parameters[target_name] = replacement

    def _tie_or_clone(
        self,
        model: Module,
        clone: bool,
    ) -> Optional[Any]:
        source_parameter = self._get_source_parameter(model)
        target_parameter = self._get_target_parameter(model)

        if not isinstance(target_parameter, SharedParameterPlaceholder):
            raise ValueError(
                f"Shared parameter '{self.target}' expected to be of type "
                f"`{SharedParameterPlaceholder.__name__}`, but got `{type(target_parameter).__name__}`"
            )
        assert isinstance(target_parameter, SharedParameterPlaceholder)
        self._validate_shareability(source_parameter)

        if source_parameter.shape != target_parameter.wrapped_shape:
            raise ValueError(
                f"Mismatching shapes ({source_parameter.shape} != {target_parameter.wrapped_shape}) "
                f"for shared parameters '{self.source}' and '{self.target}'"
            )
        elif source_parameter.dtype != target_parameter.wrapped_dtype:
            raise ValueError(
                f"Mismatching dtypes ({source_parameter.dtype} != {target_parameter.wrapped_dtype}) "
                f"for shared parameters '{self.source}' and '{self.target}'"
            )

        if clone:
            self._replace_target_parameter(
                model, Parameter(source_parameter.clone().detach())
            )
            return None
        else:
            self._replace_target_parameter(model, source_parameter)
            return target_parameter

    def initialize(self, model: Module):
        target_parameter = self._get_target_parameter(model)
        self._validate_shareability(target_parameter)

        placeholder = SharedParameterPlaceholder.for_param(target_parameter)
        self._replace_target_parameter(model, placeholder)

    def tie(self, model: Module) -> Optional[Any]:
        return self._tie_or_clone(model, clone=False)

    def untie(
        self,
        model: Module,
        metadata: Optional[Any] = None,
    ):
        target_parameter = self._get_target_parameter(model)
        if isinstance(target_parameter, SharedParameterPlaceholder):
            raise ValueError(
                f"Shared parameter '{self.target}' was not tied before untying"
            )

        assert isinstance(metadata, SharedParameterPlaceholder)
        self._replace_target_parameter(model, metadata)

    def clone(self, model: Module):
        self._tie_or_clone(model, clone=True)
