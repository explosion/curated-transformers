import pytest
import torch

from curated_transformers._compat import has_hf_transformers
from curated_transformers.tokenization import PiecesWithIds
from curated_transformers.tokenization import BertTokenizer
from curated_transformers.tokenization.bert_tokenizer import BertPreEncoder
from curated_transformers.tokenization.chunks import (
    InputChunks,
    SpecialPieceChunk,
    TextChunk,
)

from .util import compare_tokenizer_outputs_with_hf_tokenizer
from ..util import torch_assertclose


@pytest.fixture
def toy_tokenizer_from_files(test_dir):
    return BertTokenizer.from_files(
        vocab_path=test_dir / "toy.wordpieces",
    )


@pytest.fixture
def toy_tokenizer_from_tokenizer_json(test_dir):
    return BertTokenizer.from_tokenizer_json_file(test_dir / "toy-bert.json")


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "bert-base-cased", BertTokenizer
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_tokenizer_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts,
        "bert-base-cased",
        BertTokenizer,
        from_hf_tokenizer=True,
        # Use a revision from before tokenizer.json.
        revision="6c38be42d6b3ed125933ca95fd0165fb9df8e414",
    )


def test_from_json_file(toy_tokenizer_from_tokenizer_json, short_sample_texts):
    encoding = toy_tokenizer_from_tokenizer_json(short_sample_texts)
    _check_toy_tokenizer(encoding)

    assert toy_tokenizer_from_tokenizer_json.decode(encoding.ids) == [
        "I saw a girl with a telescope.",
        "Today we will eat pok [UNK] bowl, lots of it!",
    ]


def test_from_files(toy_tokenizer_from_files, short_sample_texts):
    encoding = toy_tokenizer_from_files(short_sample_texts)
    _check_toy_tokenizer(encoding)

    assert toy_tokenizer_from_files.decode(encoding.ids) == [
        "I saw a girl with a telescope.",
        "Today we will eat pok [UNK] bowl, lots of it!",
    ]


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 2
    assert len(pieces.pieces) == 2

    assert pieces.ids == [
        [2, 41, 818, 61, 67, 193, 88, 204, 61, 251, 909, 682, 102, 95, 17, 3],
        [
            2,
            824,
            98,
            189,
            311,
            417,
            65,
            155,
            503,
            99,
            1,
            416,
            117,
            88,
            15,
            844,
            91,
            100,
            163,
            183,
            5,
            3,
        ],
    ]
    assert pieces.pieces == [
        [
            "[CLS]",
            "I",
            "saw",
            "a",
            "g",
            "##ir",
            "##l",
            "with",
            "a",
            "te",
            "##les",
            "##co",
            "##p",
            "##e",
            ".",
            "[SEP]",
        ],
        [
            "[CLS]",
            "To",
            "##d",
            "##ay",
            "we",
            "will",
            "e",
            "##at",
            "po",
            "##k",
            "[UNK]",
            "bo",
            "##w",
            "##l",
            ",",
            "lo",
            "##t",
            "##s",
            "of",
            "it",
            "!",
            "[SEP]",
        ],
    ]

    torch_assertclose(
        pieces.padded_tensor(padding_id=1),
        torch.tensor(
            [
                [
                    2,
                    41,
                    818,
                    61,
                    67,
                    193,
                    88,
                    204,
                    61,
                    251,
                    909,
                    682,
                    102,
                    95,
                    17,
                    3,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    2,
                    824,
                    98,
                    189,
                    311,
                    417,
                    65,
                    155,
                    503,
                    99,
                    1,
                    416,
                    117,
                    88,
                    15,
                    844,
                    91,
                    100,
                    163,
                    183,
                    5,
                    3,
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
                ],
            ]
        ),
    )


def test_bert_tokenizer_preencoder():
    preencoder = BertPreEncoder(
        lowercase=False, strip_accents=True, bos_piece="[CLS]", eos_piece="[SEP]"
    )
    assert preencoder([InputChunks([TextChunk("AWO-Mitarbeiter")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk("AWO - Mitarbeiter"),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
    assert preencoder([InputChunks([TextChunk("-Mitarbeiter")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk("- Mitarbeiter"),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
    assert preencoder([InputChunks([TextChunk("AWO-")])]) == [
        InputChunks(
            [SpecialPieceChunk("[CLS]"), TextChunk("AWO -"), SpecialPieceChunk("[SEP]")]
        )
    ]
    assert preencoder([InputChunks([TextChunk("-")])]) == [
        InputChunks(
            [SpecialPieceChunk("[CLS]"), TextChunk("-"), SpecialPieceChunk("[SEP]")]
        )
    ]
    assert preencoder([InputChunks([TextChunk("")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk(""),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
    assert preencoder([InputChunks([TextChunk("Brötchen")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk("Brotchen"),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
    assert preencoder([InputChunks([TextChunk("Mw.-St.")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk("Mw . - St ."),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
