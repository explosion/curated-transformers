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
        "Rust is a multi-paradigm, high-level, general-purpose programming language. Rust is designed to have a small, stable, and fast implementation. Rust is also designed to be easy to learn. Rust is intended to be used for writing fast, reliable, and portable code.\n\n",
        "SpaCy is an open-source natural language processing (NLP) library for Python. It is designed to be fast, scalable, and easy to use.\n\n",
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
    answers = [
        "Rust is a multi-paradigm,",
        "SpaCy is an open-source natural language",
    ]
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
        "SpaCy (short for Spanish-Portuguese-Chinese artificial intelligence language model) is an open-source language model (LLM) that uses a novel combination of deep learning, big-data, and semantic parsing to provide the best accuracy available with no human involvement. It is one of the most effective open-source language models.\n\n",
        "SpaCy is an open-source package for Python that provides language model (LM) and automatic speech recognition (ASR) functionality.\n\n",
    ]

    torch.manual_seed(0)
    assert dolly_generator(
        prompts, config=SampleGeneratorConfig(top_k=5, temperature=2)
    ) == [
        "SpaCy (short for spaCoN-based AI for Cyber-Security) is open source software that is used for automated language and named entity recognition (NER). SpaCy has been specifically developed for cybersecurity text processing and uses techniques that were developed for NLP over many years by the University of Pennsylvania's Named Entity Tagging (NetNesp) project, by Penn Tree Street Labs and other contributors, with a focus towards named- Entity Resolution (ner).\n\n",
        "The spaCy library is an NLP package based on Stanford's SpanTweaked architecture for Stanford CoreNLP. SpaCy's main difference over Stanford CoreNLP is the usage of the PySpREnd language model, which was designed for Natural Language Processings, rather than just a simple parser, like the original SpanTweaked. This makes it possible to do much better NLP with the model than just the simple parsing of Stanford's ParserCore. However, spaCy is also a more general-purpose library, which also allows for more complex NLP tasks than just sentence parsing, such as part-of-speech taggers and dependency parsing, so it can be a good starting point for a NLTK-style library. In the field of machine learning and artificial intelligence, a spa file (spatial Pyramids file) describes a specific model trained on data, such as a part-of-speech tagged corpus, and it can be loaded and executed with spaCy's model or the Stanford CoreNLP's models (see Stanford CoreNLIps's API for loading a model from a spaCy spa file).\n\n",
    ]
