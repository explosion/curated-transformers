import pytest
import torch

from curated_transformers._compat import has_bitsandbytes
from curated_transformers.generation.config import GreedyGeneratorConfig
from curated_transformers.generation.dolly_v2 import DollyV2Generator
from curated_transformers.quantization import BitsAndBytesConfig

from ..conftest import GPU_TESTS_ENABLED


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


@pytest.mark.veryslow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
@pytest.mark.skipif(not has_bitsandbytes, reason="requires bitsandbytes")
def test_8_bit_quantization(dolly_generator_8_bit):
    prompts = [
        "What is the Rust programming language?",
        "What is spaCy?",
    ]
    expected = [
        "Rust is a multi-paradigm, high-level, general-purpose programming language. Rust is designed to have a small, consistent, and predictable set of language features that together provide a high-level of productivity and reliability. Rust's",
        "SpaCy is an open-source library for natural language processing (NLP) and language understanding (LU) on the web. It is designed to be fast, scalable, and easy to use.\n\n",
    ]
    generated = dolly_generator_8_bit(
        prompts, config=GreedyGeneratorConfig(max_generated_pieces=50)
    )
    assert generated == expected

    prompts = [
        "What is spaCy?",
        "What is the Rust programming language?",
    ]
    expected = expected[::-1]
    generated = dolly_generator_8_bit(
        prompts, config=GreedyGeneratorConfig(max_generated_pieces=50)
    )
    assert generated == expected


@pytest.mark.veryslow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
@pytest.mark.skipif(not has_bitsandbytes, reason="requires bitsandbytes")
def test_4_bit_quantization(dolly_generator_4_bit):
    prompts = [
        "What is the Rust programming language?",
        "What is spaCy?",
    ]
    expected = [
        "Rust is a multi-paradigm, high-level, general-purpose programming language. Rust is a compiled language, but unlike other compiled languages, Rust guarantees memory safety, which means that all memory accesses are guaranteed to be valid.",
        "SpaCy is an NLP library for the Python programming language. It is developed by the spacy team at IST Austria.\n\n",
    ]
    generated = dolly_generator_4_bit(
        prompts, config=GreedyGeneratorConfig(max_generated_pieces=50)
    )
    assert generated == expected

    prompts = [
        "What is spaCy?",
        "What is the Rust programming language?",
    ]
    expected = expected[::-1]
    generated = dolly_generator_4_bit(
        prompts, config=GreedyGeneratorConfig(max_generated_pieces=50)
    )
    assert generated == expected


@pytest.mark.veryslow
@pytest.mark.skipif(not has_bitsandbytes, reason="requires bitsandbytes")
def test_on_non_gpu():
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
