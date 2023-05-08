import pytest
import torch

from curated_transformers._compat import has_hf_transformers
from curated_transformers.tokenization import ByteBPETokenizer, PiecesWithIds

from ..util import torch_assertclose


@pytest.fixture
def toy_tokenizer(test_dir):
    return ByteBPETokenizer.from_files(
        vocab_path=test_dir / "toy-vocab.json", merges_path=test_dir / "toy-merges.txt"
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.slow
def test_bbpe_encoder_hf_model(sample_texts):
    tokenizer = ByteBPETokenizer.from_hf_hub(name="roberta-base")
    pieces = tokenizer(sample_texts)
    _check_roberta_base_tokenizer(pieces)


def test_bbpe_encoder(toy_tokenizer, sample_texts):
    encoding = toy_tokenizer(sample_texts)
    _check_toy_tokenizer(encoding)


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 2
    assert len(pieces.pieces) == 2

    assert pieces.lens == [
        [1, 1, 1, 1, 3, 1, 1, 7, 1],
        [1, 3, 1, 1, 2, 4, 4, 3, 1, 2, 1],
    ]
    assert pieces.ids == [
        [
            0,
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
            2,
        ],
        [
            0,
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
            2,
        ],
    ]
    assert pieces.pieces == [
        [
            "<s>",
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
            "</s>",
        ],
        [
            "<s>",
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
            "</s>",
        ],
    ]

    torch_assertclose(
        pieces.padded_tensor,
        torch.tensor(
            [
                [
                    0,
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
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    0,
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
                    2,
                ],
            ],
            dtype=torch.int32,
        ),
    )
    torch_assertclose(
        pieces.attention_mask,
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
                    True,
                    True,
                ],
            ]
        ),
    )


def _check_roberta_base_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert pieces.lens == [
        [1, 1, 1, 1, 1, 1, 1, 2, 1],
        [1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1],
    ]
    assert pieces.ids == [
        [0, 100, 794, 10, 1816, 19, 10, 27608, 4, 2],
        [0, 5625, 52, 40, 3529, 181, 48344, 5749, 6, 3739, 9, 24, 328, 2],
    ]
    assert pieces.pieces == [
        ["<s>", "I", "Ġsaw", "Ġa", "Ġgirl", "Ġwith", "Ġa", "Ġtelescope", ".", "</s>"],
        [
            "<s>",
            "Today",
            "Ġwe",
            "Ġwill",
            "Ġeat",
            "Ġp",
            "okÃ©",
            "Ġbowl",
            ",",
            "Ġlots",
            "Ġof",
            "Ġit",
            "!",
            "</s>",
        ],
    ]
    torch_assertclose(
        pieces.padded_tensor,
        torch.tensor(
            [
                [0, 100, 794, 10, 1816, 19, 10, 27608, 4, 2, 1, 1, 1, 1],
                [0, 5625, 52, 40, 3529, 181, 48344, 5749, 6, 3739, 9, 24, 328, 2],
            ],
            dtype=torch.int32,
        ),
    )
    torch_assertclose(
        pieces.attention_mask,
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
                ],
            ]
        ),
    )
