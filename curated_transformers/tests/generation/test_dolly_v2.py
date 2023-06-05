from typing import Iterator, List, Tuple
import pytest
import torch

from curated_transformers.generation.dolly_v2 import DollyV2Generator

from ..conftest import GPU_TESTS_ENABLED


def _collect_pieces(generator: Iterator[List[Tuple[int, str]]], n: int) -> List[str]:
    pieces = [[] for _ in range(n)]
    for step_pieces in generator:
        for seq_id, piece in step_pieces:
            pieces[seq_id].append(piece)

    return ["".join(answer) for answer in pieces]


@pytest.mark.veryslow
@pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="requires GPU")
def test_generate_deterministic():
    generator = DollyV2Generator.from_hf_hub(
        name="databricks/dolly-v2-3b", device=torch.device("cuda", index=0)
    )

    prompts = [
        "What is the Rust programming language?",
        "What is spaCy?",
    ]
    assert _collect_pieces(generator(prompts), len(prompts)) == [
        "Rust is a multi-paradigm, high-level, general-purpose programming language. Rust is designed to have a small, stable, and fast implementation. Rust is also designed to be easy to learn. Rust is intended to be used for writing fast, reliable, and portable code.\n\n",
        "SpaCy is an open-source natural language processing (NLP) library for Python. It is designed to be fast, scalable, and easy to use.\n\n",
    ]

    prompts = [
        "What is spaCy?",
        "What is the Rust programming language?",
    ]
    assert _collect_pieces(generator(prompts), len(prompts)) == [
        "SpaCy is an open-source natural language processing (NLP) library for Python. It is designed to be fast, scalable, and easy to use.\n\n",
        "Rust is a multi-paradigm, high-level, general-purpose programming language. Rust is designed to have a small, stable, and fast implementation. Rust is also designed to be easy to learn. Rust is intended to be used for writing fast, reliable, and portable code.\n\n",
    ]
