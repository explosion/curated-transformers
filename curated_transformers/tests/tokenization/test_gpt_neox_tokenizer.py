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
    return GPTNeoXTokenizer.from_files(
        vocab_path=test_dir / "toy-vocab.json", merges_path=test_dir / "toy-merges.txt"
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.slow
def test_gptneox_tokenizer_against_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "EleutherAI/gpt-neox-20b", GPTNeoXTokenizer, pad_token="[PAD]"
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.slow
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


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.slow
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
        [
            44,
            997,
            262,
            305,
            334,
            79,
            342,
            262,
            388,
            79,
            302,
            70,
            472,
            72,
            17,
        ],
        [
            55,
            841,
            321,
            362,
            579,
            324,
            294,
            291,
            494,
            131,
            106,
            270,
            307,
            79,
            15,
            298,
            303,
            86,
            287,
            317,
            4,
        ],
    ]
    assert pieces.pieces == [
        [
            "I",
            "Ġsaw",
            "Ġa",
            "Ġg",
            "ir",
            "l",
            "Ġwith",
            "Ġa",
            "Ġte",
            "l",
            "es",
            "c",
            "op",
            "e",
            ".",
        ],
        [
            "T",
            "od",
            "ay",
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
                    44,
                    997,
                    262,
                    305,
                    334,
                    79,
                    342,
                    262,
                    388,
                    79,
                    302,
                    70,
                    472,
                    72,
                    17,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    55,
                    841,
                    321,
                    362,
                    579,
                    324,
                    294,
                    291,
                    494,
                    131,
                    106,
                    270,
                    307,
                    79,
                    15,
                    298,
                    303,
                    86,
                    287,
                    317,
                    4,
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
                    1,
                    44,
                    997,
                    262,
                    305,
                    334,
                    79,
                    342,
                    262,
                    388,
                    79,
                    302,
                    70,
                    472,
                    72,
                    17,
                ],
                [
                    55,
                    841,
                    321,
                    362,
                    579,
                    324,
                    294,
                    291,
                    494,
                    131,
                    106,
                    270,
                    307,
                    79,
                    15,
                    298,
                    303,
                    86,
                    287,
                    317,
                    4,
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
                    False,
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
