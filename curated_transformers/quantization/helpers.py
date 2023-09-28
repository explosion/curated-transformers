from typing import Optional

from torch.nn import Module

from ..util.serde.load import TensorToParameterConverterT
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
