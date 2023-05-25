import pytest
import torch

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.tokenization import GPTNeoXTokenizer, PiecesWithIds
from curated_transformers.tokenization.chunks import (
    InputChunks,
    SpecialPieceChunk,
    TextChunk,
)

from .util import compare_tokenizer_outputs_with_hf_tokenizer
from ..util import torch_assertclose


@pytest.fixture
def toy_tokenizer(test_dir):
    return GPTNeoXTokenizer.from_tokenizer_json_file(test_dir / "toy-gpt-neox.json")


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "EleutherAI/gpt-neox-20b", GPTNeoXTokenizer, pad_token="[PAD]"
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_tokenizer_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts,
        "EleutherAI/gpt-neox-20b",
        GPTNeoXTokenizer,
        pad_token="[PAD]",
        from_hf_tokenizer=True,
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_gptneox_tokenizer_against_hf_tokenizer_special_tokens():
    texts_with_special_tokens = [
        "### Instruction: foo ### Response: bar ### End",
        "### Instruction: foo ### Response: bar",
        "### Instruction:### Response:### End",
    ]

    chunks_with_special_tokens = [
        InputChunks(
            [
                SpecialPieceChunk("### Instruction:", after=" "),
                TextChunk("foo"),
                SpecialPieceChunk("### Response:", after=" ", before=" "),
                TextChunk("bar"),
                SpecialPieceChunk("### End", before=" "),
            ]
        ),
        InputChunks(
            [
                SpecialPieceChunk("### Instruction:", after=" "),
                TextChunk("foo"),
                SpecialPieceChunk("### Response:", after=" ", before=" "),
                TextChunk("bar"),
            ]
        ),
        InputChunks(
            [
                SpecialPieceChunk("### Instruction:"),
                SpecialPieceChunk("### Response:"),
                SpecialPieceChunk("### End"),
            ]
        ),
    ]

    tokenizer = GPTNeoXTokenizer.from_hf_hub(name="databricks/dolly-v2-3b")
    pieces = tokenizer(chunks_with_special_tokens)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
    hf_pieces = hf_tokenizer(texts_with_special_tokens)

    assert pieces.ids == hf_pieces["input_ids"]


def test_gptneox_tokenizer_roundtrip(sample_texts):
    tokenizer = GPTNeoXTokenizer.from_hf_hub(name="EleutherAI/gpt-neox-20b")
    pieces = tokenizer(sample_texts)
    decoded = tokenizer.decode(pieces.ids)

    assert decoded == sample_texts


def test_gpt_neox_toy_tokenizer(toy_tokenizer, short_sample_texts):
    encoding = toy_tokenizer(short_sample_texts)
    _check_toy_tokenizer(encoding)


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 2
    assert len(pieces.pieces) == 2

    assert pieces.ids == [
        [42, 396, 88, 260, 304, 329, 77, 346, 260, 457, 295, 84, 68, 481, 70, 15],
        [
            53,
            80,
            840,
            363,
            595,
            326,
            289,
            292,
            489,
            129,
            104,
            267,
            301,
            77,
            13,
            307,
            297,
            84,
            285,
            314,
            2,
        ],
    ]

    assert pieces.pieces == [
        [
            "I",
            "Ġsa",
            "w",
            "Ġa",
            "Ġg",
            "ir",
            "l",
            "Ġwith",
            "Ġa",
            "Ġte",
            "le",
            "s",
            "c",
            "op",
            "e",
            ".",
        ],
        [
            "T",
            "o",
            "day",
            "Ġwe",
            "Ġwill",
            "Ġe",
            "at",
            "Ġp",
            "ok",
            "Ã",
            "©",
            "Ġb",
            "ow",
            "l",
            ",",
            "Ġl",
            "ot",
            "s",
            "Ġof",
            "Ġit",
            "!",
        ],
    ]

    torch_assertclose(
        pieces.padded_tensor(padding_id=1),
        torch.tensor(
            [
                [
                    42,
                    396,
                    88,
                    260,
                    304,
                    329,
                    77,
                    346,
                    260,
                    457,
                    295,
                    84,
                    68,
                    481,
                    70,
                    15,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    53,
                    80,
                    840,
                    363,
                    595,
                    326,
                    289,
                    292,
                    489,
                    129,
                    104,
                    267,
                    301,
                    77,
                    13,
                    307,
                    297,
                    84,
                    285,
                    314,
                    2,
                ],
            ],
            dtype=torch.int32,
        ),
    )
    torch_assertclose(
        pieces.padded_tensor(padding_id=1, pad_left=True),
        torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    42,
                    396,
                    88,
                    260,
                    304,
                    329,
                    77,
                    346,
                    260,
                    457,
                    295,
                    84,
                    68,
                    481,
                    70,
                    15,
                ],
                [
                    53,
                    80,
                    840,
                    363,
                    595,
                    326,
                    289,
                    292,
                    489,
                    129,
                    104,
                    267,
                    301,
                    77,
                    13,
                    307,
                    297,
                    84,
                    285,
                    314,
                    2,
                ],
            ],
            dtype=torch.int32,
        ),
    )
    torch_assertclose(
        pieces.attention_mask(),
        torch.tensor(
            [
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            ]
        ),
    )

    torch_assertclose(
        pieces.attention_mask(pad_left=True),
        torch.tensor(
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            ]
        ),
    )
