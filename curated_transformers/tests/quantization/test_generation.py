from copy import deepcopy

import pytest
import torch

from curated_transformers._compat import has_bitsandbytes
from curated_transformers.generation.config import GreedyGeneratorConfig
from curated_transformers.generation.dolly_v2 import DollyV2Generator
from curated_transformers.quantization.bnb import BitsAndBytesConfig

from ..conftest import GPU_TESTS_ENABLED

PROMPTS_AND_KEYWORDS = [
    [
        "What is the Rust programming language?",
        "What is spaCy?",
    ],
    [
        ["Rust", "multi-paradigm", "language", "design"],
        ["SpaCy", "NLP", "library", "Python"],
    ],
]


@pytest.fixture(scope="module")
def dolly_generator_8_bit():
    return DollyV2Generator.from_hf_hub(
        name="databricks/dolly-v2-3b",
        device=torch.device("cuda", index=0),
        quantization_config=BitsAndBytesConfig.for_8bit(),
    )


@pytest.fixture(scope="module")
def dolly_generator_4_bit():
    return DollyV2Generator.from_hf_hub(
        name="databricks/dolly-v2-3b",
        device=torch.device("cuda", index=0),
        quantization_config=BitsAndBytesConfig.for_4bit(),
    )


def _check_quantized_generator_output(output, expected_keywords):
    # Due the inherent non-determinism of executing low-bit quantized modules on
    # different GPUs (which can use different kernels), we can't reliably expect the
    # output to match a string verbatim. So, we'll just look for specific, low-frequency
    # keywords as a way to detect if gibberish/irrelevant text is being generated.
    for output, keywords in zip(output, expected_keywords):
        assert all(keyword in output for keyword in keywords)


@pytest.mark.slow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
@pytest.mark.skipif(not has_bitsandbytes, reason="requires bitsandbytes")
def test_quantization_8_bit(dolly_generator_8_bit):
    prompts = deepcopy(PROMPTS_AND_KEYWORDS[0])
    expected = deepcopy(PROMPTS_AND_KEYWORDS[1])
    generated = dolly_generator_8_bit(
        prompts, config=GreedyGeneratorConfig(max_generated_pieces=50)
    )
    _check_quantized_generator_output(generated, expected)

    prompts = prompts[::-1]
    expected = expected[::-1]
    generated = dolly_generator_8_bit(
        prompts, config=GreedyGeneratorConfig(max_generated_pieces=50)
    )
    _check_quantized_generator_output(generated, expected)


@pytest.mark.slow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
@pytest.mark.skipif(not has_bitsandbytes, reason="requires bitsandbytes")
def test_quantization_4_bit(dolly_generator_4_bit):
    prompts = deepcopy(PROMPTS_AND_KEYWORDS[0])
    expected = deepcopy(PROMPTS_AND_KEYWORDS[1])
    generated = dolly_generator_4_bit(
        prompts, config=GreedyGeneratorConfig(max_generated_pieces=50)
    )
    _check_quantized_generator_output(generated, expected)

    prompts = prompts[::-1]
    expected = expected[::-1]
    generated = dolly_generator_4_bit(
        prompts, config=GreedyGeneratorConfig(max_generated_pieces=50)
    )
    _check_quantized_generator_output(generated, expected)


@pytest.mark.slow
@pytest.mark.skipif(not has_bitsandbytes, reason="requires bitsandbytes")
def test_quantization_on_non_gpu():
    with pytest.raises(ValueError, match="only be performed on CUDA"):
        model = DollyV2Generator.from_hf_hub(
            name="databricks/dolly-v2-3b",
            device=None,
            quantization_config=BitsAndBytesConfig.for_8bit(),
        )

    with pytest.raises(ValueError, match="only be performed on CUDA"):
        model = DollyV2Generator.from_hf_hub(
            name="databricks/dolly-v2-3b",
            device=torch.device("cpu"),
            quantization_config=BitsAndBytesConfig.for_4bit(),
        )
