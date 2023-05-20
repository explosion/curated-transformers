import pytest
import torch

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.tokenization import PiecesWithIds
from curated_transformers.tokenization import BertTokenizer
from curated_transformers.tokenization.xlmr_tokenizer import XlmrTokenizer

from ..util import torch_assertclose


@pytest.fixture
def toy_tokenizer(test_dir):
    return XlmrTokenizer.from_files(
        model_path=test_dir / "toy.model",
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.slow
def test_xlmrtokenizer_hf_tokenizer(sample_texts):
    tokenizer = XlmrTokenizer.from_hf_hub(name="xlm-roberta-base")
    pieces = tokenizer(sample_texts)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    hf_pieces = hf_tokenizer(sample_texts)

    assert pieces.ids == hf_pieces["input_ids"]


def test_xlmr_toy_tokenizer(toy_tokenizer, short_sample_texts):
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

    assert pieces.lens == [
        [1, 1, 1, 1, 1, 1, 1, 7, 1],
        [1, 2, 1, 1, 1, 4, 4, 3, 1, 2, 1],
    ]
    assert pieces.ids == [
        [0, 9, 466, 11, 948, 42, 11, 171, 169, 111, 29, 21, 144, 5, 2],
        [
            0,
            484,
            547,
            113,
            172,
            568,
            63,
            21,
            46,
            3,
            85,
            116,
            28,
            4,
            149,
            227,
            7,
            14,
            26,
            147,
            2,
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
                    0,
                    9,
                    466,
                    11,
                    948,
                    42,
                    11,
                    171,
                    169,
                    111,
                    29,
                    21,
                    144,
                    5,
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
                    484,
                    547,
                    113,
                    172,
                    568,
                    63,
                    21,
                    46,
                    3,
                    85,
                    116,
                    28,
                    4,
                    149,
                    227,
                    7,
                    14,
                    26,
                    147,
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
