import pytest
import torch

from curated_transformers.tokenizers import PiecesWithIds
from curated_transformers.tokenizers.legacy.camembert_tokenizer import (
    CamemBERTTokenizer,
)
from curated_transformers.util.serde import LocalModelFile

from ...compat import has_hf_transformers
from ...utils import torch_assertclose
from ..util import compare_tokenizer_outputs_with_hf_tokenizer


@pytest.fixture
def toy_tokenizer(test_dir):
    return CamemBERTTokenizer.from_files(
        model_file=LocalModelFile(path=test_dir / "toy.model"),
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer(sample_texts, french_sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "camembert-base", CamemBERTTokenizer
    )
    compare_tokenizer_outputs_with_hf_tokenizer(
        french_sample_texts, "camembert-base", CamemBERTTokenizer
    )


def test_camembert_tokenizer_toy_tokenizer(toy_tokenizer, short_sample_texts):
    encoding = toy_tokenizer(short_sample_texts)
    _check_toy_tokenizer(encoding)

    decoded = toy_tokenizer.decode(encoding.ids)
    assert decoded == [
        "I saw a girl with a telescope.",
        "Today we will eat pok ⁇  bowl, lots of it!",
        "Tokens which are unknown in ⁇  most ⁇  latin ⁇  alphabet ⁇  vocabularies.",
    ]


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 3
    assert len(pieces.pieces) == 3

    assert pieces.ids == [
        [5, 12, 469, 14, 951, 45, 14, 174, 172, 114, 32, 24, 147, 8, 6],
        [
            5,
            487,
            550,
            116,
            175,
            571,
            66,
            24,
            49,
            3,
            88,
            119,
            31,
            7,
            152,
            230,
            10,
            17,
            29,
            150,
            6,
        ],
        [
            5,
            487,
            49,
            98,
            10,
            143,
            126,
            225,
            49,
            28,
            119,
            28,
            23,
            3,
            640,
            3,
            152,
            80,
            57,
            3,
            14,
            31,
            33,
            56,
            22,
            69,
            19,
            18,
            3,
            11,
            87,
            24,
            32,
            22,
            69,
            235,
            53,
            461,
            8,
            6,
        ],
    ]

    assert pieces.pieces == [
        [
            "<s>",
            "▁I",
            "▁saw",
            "▁a",
            "▁girl",
            "▁with",
            "▁a",
            "▁t",
            "el",
            "es",
            "c",
            "o",
            "pe",
            ".",
            "</s>",
        ],
        [
            "<s>",
            "▁To",
            "day",
            "▁we",
            "▁will",
            "▁eat",
            "▁p",
            "o",
            "k",
            "é",
            "▁b",
            "ow",
            "l",
            ",",
            "▁l",
            "ot",
            "s",
            "▁of",
            "▁it",
            "!",
            "</s>",
        ],
        [
            "<s>",
            "▁To",
            "k",
            "en",
            "s",
            "▁which",
            "▁are",
            "▁un",
            "k",
            "n",
            "ow",
            "n",
            "▁in",
            "ペ",
            "▁most",
            "で",
            "▁l",
            "at",
            "in",
            "が",
            "▁a",
            "l",
            "p",
            "h",
            "a",
            "b",
            "e",
            "t",
            "際",
            "▁",
            "v",
            "o",
            "c",
            "a",
            "b",
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
                    5,
                    12,
                    469,
                    14,
                    951,
                    45,
                    14,
                    174,
                    172,
                    114,
                    32,
                    24,
                    147,
                    8,
                    6,
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
                    5,
                    487,
                    550,
                    116,
                    175,
                    571,
                    66,
                    24,
                    49,
                    3,
                    88,
                    119,
                    31,
                    7,
                    152,
                    230,
                    10,
                    17,
                    29,
                    150,
                    6,
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
                    5,
                    487,
                    49,
                    98,
                    10,
                    143,
                    126,
                    225,
                    49,
                    28,
                    119,
                    28,
                    23,
                    3,
                    640,
                    3,
                    152,
                    80,
                    57,
                    3,
                    14,
                    31,
                    33,
                    56,
                    22,
                    69,
                    19,
                    18,
                    3,
                    11,
                    87,
                    24,
                    32,
                    22,
                    69,
                    235,
                    53,
                    461,
                    8,
                    6,
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
                ],
            ]
        ),
    )
