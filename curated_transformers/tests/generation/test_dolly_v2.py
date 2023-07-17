import pytest
import torch

from curated_transformers.generation.config import (
    GreedyGeneratorConfig,
    SampleGeneratorConfig,
)
from curated_transformers.generation.dolly_v2 import DollyV2Generator

from ..conftest import GPU_TESTS_ENABLED


@pytest.fixture(scope="module")
def dolly_generator():
    return DollyV2Generator.from_hf_hub(
        name="databricks/dolly-v2-3b", device=torch.device("cuda", index=0)
    )


@pytest.mark.slow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
def test_generate_deterministic(dolly_generator):
    prompts = [
        "What is the Rust programming language?",
        "What is spaCy?",
    ]
    answers = [
        "Rust is a multi-paradigm, high-level, general-purpose programming language. Rust is designed to have a small, consistent, and predictable language surface. Rust is also designed to be efficient, and to have a small memory footprint. Rust is designed to be safe, and to have a well-defined memory model. Rust is also designed to be concurrent, and to have a good support for concurrent programming. Rust is designed to be fast, and to have a good support for performance-critical code. Rust is also designed to be modular, and to have a good support for modular programming. Rust is designed to have a good support for internationalization and localization.\n\n",
        "SpaCy is a natural language processing (NLP) library for Python that provides tokenization, part-of-speech (POS) tagging, named entity recognition (NER), and dependency parsing.\n\n",
    ]
    assert dolly_generator(prompts, config=GreedyGeneratorConfig()) == answers

    # Test in reverse order to verify that sequence identifiers are
    # handled correctly.
    assert (
        dolly_generator(prompts[::-1], config=GreedyGeneratorConfig()) == answers[::-1]
    )


@pytest.mark.slow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
def test_generate_max_generated_pieces(dolly_generator):
    prompts = [
        "What is the Rust programming language?",
        "What is spaCy?",
    ]
    answers = ["Rust is a multi-paradigm,", "SpaCy is a natural language processing (N"]

    assert (
        dolly_generator(prompts, config=GreedyGeneratorConfig(max_generated_pieces=10))
        == answers
    )

    # Test in reverse order to verify that sequence identifiers are
    # handled correctly.
    assert (
        dolly_generator(
            prompts[::-1], config=GreedyGeneratorConfig(max_generated_pieces=10)
        )
        == answers[::-1]
    )


@pytest.mark.slow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
def test_generate_sample(dolly_generator):
    prompts = [
        "What is spaCy?",
        "What is spaCy?",
    ]
    # Fix the seed so that we are always randomly sampling in the same way.
    torch.manual_seed(0)
    assert dolly_generator(prompts, config=SampleGeneratorConfig(top_k=10)) == [
        "SpaCy (short for spaCy Toolkit) is a Python library for Natural Language Processing (NLP) and machine translation based on the   spaCy research project. It has been open-source since June 2023 and has been used in production for NLP tasks like POS tag classification (part of OpenStax' TACL), question answering (QACQUEL), and semantic parsing (CoreNLP) as of October 2023.\n\n",
        "SpaCy is an open-source package for Python that automates part of the process of building language models (LMs), or parsers for plain text, using a technique called graph-based learning. It supports English, French, German, Dutch, Italian, Spanish, Portuguese, Polish, and Dutch, all of which are official language groups of Europe.\n\n",
    ]

    torch.manual_seed(0)
    assert dolly_generator(
        prompts, config=SampleGeneratorConfig(top_k=5, temperature=2)
    ) == [
        "SpaCy (short for Spanish Language Model) is a natural language processor (NLP) for English based on the open source spaCy project. It supports sentence detection, part-of-speech tagging and coreference chains, among other things. spaCy was originally developed at the University of Cambridge's Machine Translation Lab. Since then, it has also developed a version optimized for Python that can be installed viapip.\n\n",
        "The spaCy library is an NLP library based on Stanford's PoSpell architecture, that is designed to make NLP easy. spaCy supports all the standard NLTK pipeline stages and is able to outperform the NLTK on many Named Entity Tagging tasks, as well many others, both at test-level and at the system-wide average. It has a similar architecture to NLTK’s, but is designed from the ground up for the needs of a NLP research community rather than a production system: It has a smaller and less mature API, does away with its core tokenizer (which is notoriously hard to train), and is based on the Speller system from Stanford's CS-ADLDN program, which has been shown to significantly outperform the NLTK tokeniser in terms of both accuracy (93.7% on a test set of 20K tokens vs. NLTK's 76.3%) and efficiency.\n\n",
    ]
