from typing import TYPE_CHECKING, Callable, Optional, Set, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from ..._compat import has_bitsandbytes
from ...util.pytorch import ModuleIterator, apply_to_module
from ...util.serde.load import TensorToParameterConverterT
from .config import BitsAndBytesConfig, _4BitConfig, _8BitConfig

if TYPE_CHECKING:
    import bitsandbytes as bnb
else:
    bnb = None


def prepare_for_quantization(
    module: Module,
    config: BitsAndBytesConfig,
    non_quantizable_module_prefixes: Set[str],
) -> Optional[TensorToParameterConverterT]:
    """
    Prepares a PyTorch module for quantization using the ``bitsandbytes``
    library.

    :param module:
        Module to prepare for quantization.
    :param config:
        ``bitsandbytes`` quantization configuration.
    :param non_quantizable_module_prefixes:
        Set of module prefixes that should not be quantized.
    :returns:
        An optional callback for converting non-quantized tensors
        to parameters that are compatible with the ``bitsandbytes``
        quantization backend.
    """
    _assert_bitsandbytes_installed()

    # All inputs to quantized layers need to be of type `float16`.
    # TODO How do we exclude non-quantizable modules from this?
    module.to(torch.float16)

    if isinstance(config.inner, _8BitConfig):
        _replace_quantizable_modules(
            module, config.inner, non_quantizable_module_prefixes, _init_8bit_linear
        )
    elif isinstance(config.inner, _4BitConfig):
        _replace_quantizable_modules(
            module, config.inner, non_quantizable_module_prefixes, _init_4bit_linear
        )
    else:
        raise ValueError(f"Unknown bitsandbytes config `{type(config)}`")

    return _convert_tensor_to_quantized_parameter


def _replace_quantizable_modules(
    module: Module,
    config: Union[_8BitConfig, _4BitConfig],
    non_quantizable_module_prefixes: Set[str],
    init_quantized_module: Callable[
        [Module, Union[_8BitConfig, _4BitConfig], torch.device], Module
    ],
):
    def apply(itr: ModuleIterator):
        quantize_everything = len(non_quantizable_module_prefixes) == 0
        if not quantize_everything and itr.prefix in non_quantizable_module_prefixes:
            # Not a quantizable module.
            return
        elif itr.parent is None:
            # Root module
            return
        elif not isinstance(itr.module, torch.nn.Linear):
            # Only Linear modules can be quantized.
            return

        # Initialize on the original device, which should be the
        # `meta` device at this stage.
        device = next(itr.module.parameters()).device
        quantized_module = init_quantized_module(itr.module, config, device)
        itr.parent._modules[itr.name] = quantized_module

    apply_to_module(module, apply)


def _convert_tensor_to_quantized_parameter(
    module: Module,
    module_prefix: str,
    parameter_name: str,
    tensor: Tensor,
    device: Optional[torch.device] = None,
) -> Parameter:
    if device is None or "cuda" not in device.type:
        raise ValueError(
            f"bitsandbytes quantization can only be performed on CUDA GPU devices"
        )

    old_param = module._parameters[parameter_name]
    assert old_param is not None
    if old_param.shape != tensor.shape:
        raise ValueError(
            f"Expected size of replacement for `{module_prefix}` to be {old_param.shape}, but got {tensor.shape}"
        )

    import bitsandbytes as bnb

    # Bias is stored as a regular, non-quantized parameter.
    is_bias_param = parameter_name == "bias"
    # All parameters need to be of dtype `float16` for quantization.
    tensor = tensor.to(torch.float16)

    is_8bit = isinstance(module, bnb.nn.Linear8bitLt)
    is_4bit = isinstance(module, bnb.nn.Linear4bit)
    is_non_quantized = not is_8bit and not is_4bit

    if is_bias_param or is_non_quantized:
        new_param = Parameter(tensor, requires_grad=old_param.requires_grad)
    elif is_8bit:
        assert isinstance(old_param, bnb.nn.Int8Params)
        new_param = bnb.nn.Int8Params(
            tensor, requires_grad=False, has_fp16_weights=old_param.has_fp16_weights
        )
    elif is_4bit:
        assert isinstance(old_param, bnb.nn.Params4bit)
        new_param = bnb.nn.Params4bit(
            tensor,
            requires_grad=False,
            blocksize=old_param.blocksize,
            compress_statistics=old_param.compress_statistics,
            quant_type=old_param.quant_type,
            quant_state=old_param.quant_state,
        )

    return new_param.to(device=device)  # type:ignore


def _init_8bit_linear(
    source: Module, config: Union[_8BitConfig, _4BitConfig], device: torch.device
) -> "bnb.nn.Linear8bitLt":
    assert isinstance(config, _8BitConfig)
    import bitsandbytes as bnb

    quantized_module = bnb.nn.Linear8bitLt(
        input_features=source.in_features,
        output_features=source.out_features,
        bias=source.bias is not None,
        has_fp16_weights=config.finetunable,
        threshold=config.outlier_threshold,
        device=device,
    )
    return quantized_module


def _init_4bit_linear(
    source: Module, config: Union[_8BitConfig, _4BitConfig], device: torch.device
) -> "bnb.nn.Linear4bit":
    assert isinstance(config, _4BitConfig)
    import bitsandbytes as bnb

    quantized_module = bnb.nn.Linear4bit(
        input_features=source.in_features,
        output_features=source.out_features,
        bias=source.bias is not None,
        compute_dtype=config.compute_dtype,
        compress_statistics=config.double_quantization,
        quant_type=config.quantization_dtype.value,
        device=device,
    )
    return quantized_module


def _assert_bitsandbytes_installed():
    if not has_bitsandbytes:
        raise ValueError(
            "The `bitsandbytes` Python library is required for quantization support"
        )
