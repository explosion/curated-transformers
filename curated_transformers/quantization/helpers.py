from typing import Optional

from torch.nn import Module, Parameter

from ..util.serde import TensorToParameterConverterT
from .bnb import is_quantized_module as bnb_is_quantized_module
from .bnb import is_quantized_parameter as bnb_is_quantized_parameter
from .bnb import prepare_for_quantization as bnb_prepare_for_quantization
from .bnb.config import BitsAndBytesConfig
from .quantizable import Quantizable


def prepare_module_for_quantization(
    module: Module, config: BitsAndBytesConfig
) -> Optional[TensorToParameterConverterT]:
    """
    Prepares a module for quantiazation and returns an optional callback
    to generate quantized parameter tensors.

    :param module:
        Top-level module to quantize. Should implement ``Quantizable``.
    :param config:
        Configuration for the quantizer.
    :returns:
        An optional callable that converts a non-quantized tensor
        to a quantized parameter.
    """
    if not isinstance(module, Quantizable):
        raise ValueError(f"Module of type `{type(module)}` is not quantizable")
    qmodel: Quantizable = module
    non_quantizable_module_prefixes = qmodel.modules_to_not_quantize()

    return bnb_prepare_for_quantization(module, config, non_quantizable_module_prefixes)


def is_quantized_module(module: Module) -> bool:
    """
    Returns if the module is quantized.

    :param module:
        Module to check.
    """
    return bnb_is_quantized_module(module)


def is_quantized_parameter(param: Parameter) -> bool:
    """
    Returns if the parameter is quantized.

    :param param:
        Parameter to check.
    """
    return bnb_is_quantized_parameter(param)
