from dataclasses import dataclass
from enum import Enum
from typing import Union

import torch


class Dtype4Bit(str, Enum):
    FP4 = "fp4"
    NF4 = "nf4"


@dataclass
class _4BitConfig:
    """Config for `fp4`/`nf4` quantization."""

    quantization_dtype: Dtype4Bit
    compute_dtype: torch.dtype
    double_quantization: bool


@dataclass
class _8BitConfig:
    """Config for `int8` quantization."""

    fine_tunable: bool
    outlier_threshold: float


@dataclass
class BitsAndBytesConfig:
    """Config for quantization using `bitsandbytes`."""

    inner: Union[_4BitConfig, _8BitConfig]

    @staticmethod
    def for_8bit(outlier_threshold: float = 6.0, fine_tunable: bool = False):
        """Construct a config for `int8` quantization.

        :param outlier_threshold:
            Threshold for outlier detection during weight
            decomposition.
        :param fine_tunable:
            If the quantized model should support fine-tuning after
            quantization.
        """
        return BitsAndBytesConfig(
            _8BitConfig(fine_tunable=fine_tunable, outlier_threshold=outlier_threshold)
        )

    @staticmethod
    def for_4bit(
        quantization_dtype: Dtype4Bit = Dtype4Bit.FP4,
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quantization: bool = True,
    ):
        """Construct a config for `fp4`/`nf4` quantization.

        :param quantization_dtype:
            Data type used for storing quantized weights.
        :param compute_dtype:
            Data type used for performing computations. Supported types: `float16`, `bfloat16`, `float32`
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
