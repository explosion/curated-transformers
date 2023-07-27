from dataclasses import dataclass
from enum import Enum
from typing import Union

import torch


class Dtype4Bit(str, Enum):
    """
    Data type to use for 4-bit quantization.
    """

    #: ``FP4`` - Float 4-bit.
    FP4 = "fp4"

    #: ``NF4`` - NormalFloat 4-bit.
    NF4 = "nf4"


@dataclass
class _4BitConfig:
    """
    Configuration for ``fp4``/``nf4`` quantization.
    """

    quantization_dtype: Dtype4Bit
    compute_dtype: torch.dtype
    double_quantization: bool


@dataclass
class _8BitConfig:
    """
    Configuration for ``int8`` quantization.
    """

    finetunable: bool
    outlier_threshold: float


@dataclass
class BitsAndBytesConfig:
    """
    Configuration for quantization using the ``bitsandbytes`` library.
    """

    inner: Union[_4BitConfig, _8BitConfig]

    @staticmethod
    def for_8bit(outlier_threshold: float = 6.0, finetunable: bool = False):
        """
        Construct a configuration for ``int8`` quantization.

        :param outlier_threshold:
            Threshold for outlier detection during weight
            decomposition.
        :param finetunable:
            If the quantized model should support fine-tuning after
            quantization.
        """
        return BitsAndBytesConfig(
            _8BitConfig(finetunable=finetunable, outlier_threshold=outlier_threshold)
        )

    @staticmethod
    def for_4bit(
        quantization_dtype: Dtype4Bit = Dtype4Bit.FP4,
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quantization: bool = True,
    ):
        """
        Construct a configuration for ``fp4``/``nf4`` quantization.

        :param quantization_dtype:
            Data type used for storing quantized weights.
        :param compute_dtype:
            Data type used for performing computations.
            Supported types: ``float16``, ``bfloat16``, ``float32``.
        :param double_quantization:
            If the quantization constants should themselves be
            quantized.
        """
        supported_compute_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if compute_dtype not in supported_compute_dtypes:
            raise ValueError(
                f"Unsupported compute dtype `{compute_dtype}` for quantization, must be one of: {supported_compute_dtypes}"
            )

        return BitsAndBytesConfig(
            _4BitConfig(
                quantization_dtype=quantization_dtype,
                compute_dtype=compute_dtype,
                double_quantization=double_quantization,
            )
        )
