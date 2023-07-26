import pytest
import torch
from curated_transformers.tokenizers import PiecesWithIds
from curated_transformers.tokenizers.legacy import RoBERTaTokenizer

from ...compat import has_hf_transformers
from ...util import torch_assertclose
from ..util import compare_tokenizer_outputs_with_hf_tokenizer


@pytest.fixture
def toy_tokenizer_from_files(test_dir):
    return RoBERTaTokenizer.from_files(
        vocab_path=test_dir / "toy-vocab.json", merges_path=test_dir / "toy-merges.txt"
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "roberta-base", RoBERTaTokenizer
    )


def test_from_files(toy_tokenizer_from_files, short_sample_texts):
    encoding = toy_tokenizer_from_files(short_sample_texts)
    _check_toy_tokenizer(encoding)


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 3
    assert len(pieces.pieces) == 3

    assert pieces.ids == [
        [0, 44, 997, 262, 305, 334, 79, 342, 262, 388, 79, 302, 70, 472, 72, 17, 2],
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
        [
            0,
            55,
            494,
            283,
            86,
            513,
            468,
            506,
            78,
            81,
            428,
            297,
            163,
            229,
            252,
            697,
            346,
            163,
            227,
            104,
            298,
            294,
            264,
            163,
            227,
            238,
            490,
            83,
            276,
            69,
            323,
            169,
            252,
            253,
            884,
            70,
            772,
            480,
            295,
            911,
            17,
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
        [
            "<s>",
            "T",
            "ok",
            "en",
            "s",
            "Ġwhich",
            "Ġare",
            "Ġun",
            "k",
            "n",
            "own",
            "Ġin",
            "ã",
            "ĥ",
            "ļ",
            "Ġmo",
            "st",
            "ã",
            "ģ",
            "§",
            "Ġl",
            "at",
            "in",
            "ã",
            "ģ",
            "Į",
            "Ġal",
            "p",
            "ha",
            "b",
            "et",
            "é",
            "ļ",
            "Ľ",
            "Ġvo",
            "c",
            "ab",
            "ul",
            "ar",
            "ies",
            ".",
            "</s>",
        ],
    ]

    torch_assertclose(
        pieces.padded_tensor(padding_id=1),
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
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
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
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
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
                    494,
                    283,
                    86,
                    513,
                    468,
                    506,
                    78,
                    81,
                    428,
                    297,
                    163,
                    229,
                    252,
                    697,
                    346,
                    163,
                    227,
                    104,
                    298,
                    294,
                    264,
                    163,
                    227,
                    238,
                    490,
                    83,
                    276,
                    69,
                    323,
                    169,
                    252,
                    253,
                    884,
                    70,
                    772,
                    480,
                    295,
                    911,
                    17,
                    2,
                ],
            ],
            dtype=torch.int32,
        ),
    )
    torch_assertclose(
        pieces.attention_mask().bool_mask.squeeze(dim=(1, 2)),
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
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
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
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
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
