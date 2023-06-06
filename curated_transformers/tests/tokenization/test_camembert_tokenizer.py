import pytest
import torch

from curated_transformers._compat import has_hf_transformers
from curated_transformers.tokenization import PiecesWithIds
from curated_transformers.tokenization.camembert_tokenizer import CamembertTokenizer
from curated_transformers.tokenization.chunks import (
    InputChunks,
    SpecialPieceChunk,
    TextChunk,
)

from .util import compare_tokenizer_outputs_with_hf_tokenizer
from ..util import torch_assertclose


@pytest.fixture
def toy_tokenizer(test_dir):
    return CamembertTokenizer.from_files(
        model_path=test_dir / "toy.model",
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_tokenizer_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "camembert-base", CamembertTokenizer, from_hf_tokenizer=True
    )
    sample_texts = [
        "J'ai vu une fille avec un télescope.",
        "Aujourd'hui, nous allons manger un poké bowl.",
    ]
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "camembert-base", CamembertTokenizer, from_hf_tokenizer=True
    )


def test_camembert_tokenizer_toy_tokenizer(toy_tokenizer, short_sample_texts):
    encoding = toy_tokenizer(short_sample_texts)
    _check_toy_tokenizer(encoding)

    decoded = toy_tokenizer.decode(encoding.ids)
    assert decoded == [
        "I saw a girl with a telescope.",
        "Today we will eat pok ⁇  bowl, lots of it!",
    ]


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 2
    assert len(pieces.pieces) == 2

    assert pieces.ids == [
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


def test_camembert_chunk_validation(toy_tokenizer):
    with pytest.raises(ValueError):
        toy_tokenizer([InputChunks([TextChunk("<s>")])])

    with pytest.raises(ValueError):
        toy_tokenizer([InputChunks([TextChunk("</s>")])])

    with pytest.raises(ValueError):
        toy_tokenizer([InputChunks([TextChunk("Brötchen <unk>")])])

    with pytest.raises(ValueError):
        toy_tokenizer([InputChunks([SpecialPieceChunk("Brötchen")])])

    toy_tokenizer(
        [
            InputChunks(
                [
                    SpecialPieceChunk("<s>"),
                    TextChunk("Mw . - St ."),
                    SpecialPieceChunk("</s>"),
                ]
            )
        ]
    )
