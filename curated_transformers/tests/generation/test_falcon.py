import pytest
import torch

from curated_transformers.generation.config import (
    GreedyGeneratorConfig,
    SampleGeneratorConfig,
)
from curated_transformers.generation.falcon import FalconGenerator

from ..conftest import GPU_TESTS_ENABLED


@pytest.fixture(scope="module")
def falcon_generator():
    return FalconGenerator.from_hf_hub(
        name="tiiuae/falcon-7b-instruct", device=torch.device("cuda", index=0)
    )


@pytest.mark.slow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
def test_generate_deterministic(falcon_generator):
    prompts = [
        "What is the Rust programming language?",
        "What is spaCy?",
    ]
    answers = [
        "Rust is a programming language that is designed to be a safe, concurrent, and efficient replacement for C++. It is a statically-typed language that is designed to be memory-safe and thread-safe, making it a good choice for developing high-performance applications.",
        "spaCy is a Python library for natural language processing. It is designed to be easy to use and highly customizable, making it a great tool for developers and researchers.",
    ]
    assert falcon_generator(prompts, config=GreedyGeneratorConfig()) == answers

    # Test in reverse order to verify that sequence identifiers are
    # handled correctly.
    assert (
        falcon_generator(prompts[::-1], config=GreedyGeneratorConfig()) == answers[::-1]
    )


@pytest.mark.slow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
def test_generate_max_generated_pieces(falcon_generator):
    prompts = [
        "What is the Rust programming language?",
        "What is spaCy?",
    ]
    answers = [
        "Rust is a programming language that is designed to",
        "spaCy is a Python library for natural language",
    ]
    assert (
        falcon_generator(prompts, config=GreedyGeneratorConfig(max_generated_pieces=10))
        == answers
    )

    # Test in reverse order to verify that sequence identifiers are
    # handled correctly.
    assert (
        falcon_generator(
            prompts[::-1], config=GreedyGeneratorConfig(max_generated_pieces=10)
        )
        == answers[::-1]
    )


@pytest.mark.slow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
def test_generate_sample(falcon_generator):
    prompts = [
        "What is spaCy?",
        "What is spaCy?",
    ]
    # Fix the seed so that we are always randomly sampling in the same way.
    torch.manual_seed(0)
    assert falcon_generator(prompts, config=SampleGeneratorConfig(top_k=10)) == [
        "spaCy is a Python package for natural language processing and text analysis. It is specifically designed for text classification tasks and can be used in a variety of fields, including healthcare, finance, and marketing. spaCy's main feature is its ability to extract text and its associated entities, which are then used to perform various analyses on the text, such as sentiment analysis, named entity recognition, and entity extraction.",
        "spaCy is a library for natural language processing in Python. It's built on top of NLTK's WordNet, and uses it to create a set of spaCy-compatible data structures for representing words and phrases in text. It's also a grammar-based approach to word and phrase similarity matching, and provides an API for building custom grammars.",
    ]

    torch.manual_seed(0)
    assert falcon_generator(
        prompts, config=SampleGeneratorConfig(top_k=5, temperature=2)
    ) == [
        "spacy is a Natural Language Processing tool that can be used with many programming language to build NLP-based applications such as machine learning, sentiment analysis and chatbots, and to extract text from documents and other media. spacy uses the Stanford NLP model to learn from and generate text. This makes it one of the most popular and versatile NLP libraries available. The main features of spacy include text generation and manipulation, entity extraction, part-of-speech tagging and more.",
        "spaCy is a library for natural language processing in Scala. It's designed to be easy to use and provides a range of features including text analysis tools and pre-built models to analyze various text datasets.</s> \nCan spaCy be used to analyze text from different sources or does it only work with a specific type of text or data?</s> \nspaCy can analyze text from different sources and can be used to work with text in different languages such as German and Chinese.",
    ]
