import pytest
import torch

from curated_transformers.tokenizers import PiecesWithIds
from curated_transformers.tokenizers.chunks import (
    InputChunks,
    SpecialPieceChunk,
    TextChunk,
)
from curated_transformers.tokenizers.legacy.bert_tokenizer import (
    BERTPreEncoder,
    BERTTokenizer,
)
from curated_transformers.tokenizers.legacy.legacy_tokenizer import DefaultNormalizer

from ...compat import has_hf_transformers
from ...utils import torch_assertclose
from ..util import compare_tokenizer_outputs_with_hf_tokenizer


@pytest.fixture
def toy_tokenizer_from_files(test_dir):
    return BERTTokenizer.from_files(
        vocab_file=test_dir / "toy.wordpieces",
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "bert-base-cased", BERTTokenizer
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer_short(short_sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        short_sample_texts, "bert-base-cased", BERTTokenizer
    )


def test_from_files(toy_tokenizer_from_files, short_sample_texts):
    encoding = toy_tokenizer_from_files(short_sample_texts)
    _check_toy_tokenizer(encoding)

    assert toy_tokenizer_from_files.decode(encoding.ids) == [
        "I saw a girl with a telescope.",
        "Today we will eat bowl, lots of it!",
        "Tokens which are unknown alphabet vocabularies.",
    ]

    assert toy_tokenizer_from_files.decode(encoding.ids, skip_special_pieces=False) == [
        "I saw a girl with a telescope.",
        "Today we will eat [UNK] bowl, lots of it!",
        "Tokens which are unknown [UNK] [UNK] [UNK] alphabet [UNK] vocabularies.",
    ]


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 3
    assert len(pieces.pieces) == 3

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
        [
            2,
            576,
            159,
            100,
            365,
            319,
            356,
            99,
            93,
            281,
            1,
            1,
            1,
            340,
            102,
            103,
            608,
            184,
            1,
            809,
            90,
            608,
            328,
            162,
            742,
            17,
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
        [
            "[CLS]",
            "Tok",
            "##en",
            "##s",
            "which",
            "are",
            "un",
            "##k",
            "##n",
            "##own",
            "[UNK]",
            "[UNK]",
            "[UNK]",
            "al",
            "##p",
            "##h",
            "##ab",
            "##et",
            "[UNK]",
            "vo",
            "##c",
            "##ab",
            "##ul",
            "##ar",
            "##ies",
            ".",
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
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    2,
                    576,
                    159,
                    100,
                    365,
                    319,
                    356,
                    99,
                    93,
                    281,
                    1,
                    1,
                    1,
                    340,
                    102,
                    103,
                    608,
                    184,
                    1,
                    809,
                    90,
                    608,
                    328,
                    162,
                    742,
                    17,
                    3,
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
                ],
            ]
        ),
    )


def test_bert_tokenizer_normalizer_preencoder():
    normalizer = DefaultNormalizer(lowercase=False, strip_accents=True)
    preencoder = BERTPreEncoder(bos_piece="[CLS]", eos_piece="[SEP]")

    def apply(input):
        return preencoder(normalizer(input))

    assert apply([InputChunks([TextChunk("AWO-Mitarbeiter")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk("AWO - Mitarbeiter"),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
    assert apply([InputChunks([TextChunk("-Mitarbeiter")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk("- Mitarbeiter"),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
    assert apply([InputChunks([TextChunk("AWO-")])]) == [
        InputChunks(
            [SpecialPieceChunk("[CLS]"), TextChunk("AWO -"), SpecialPieceChunk("[SEP]")]
        )
    ]
    assert apply([InputChunks([TextChunk("-")])]) == [
        InputChunks(
            [SpecialPieceChunk("[CLS]"), TextChunk("-"), SpecialPieceChunk("[SEP]")]
        )
    ]
    assert apply([InputChunks([TextChunk("")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk(""),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
    assert apply([InputChunks([TextChunk("Brötchen")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk("Brotchen"),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
    assert apply([InputChunks([TextChunk("Mw.-St.")])]) == [
        InputChunks(
            [
                SpecialPieceChunk("[CLS]"),
                TextChunk("Mw . - St ."),
                SpecialPieceChunk("[SEP]"),
            ]
        )
    ]
