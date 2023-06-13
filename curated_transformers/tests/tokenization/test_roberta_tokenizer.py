import pytest
import torch

from curated_transformers._compat import has_hf_transformers
from curated_transformers.tokenization import PiecesWithIds, RobertaTokenizer

from ..util import torch_assertclose
from .util import compare_tokenizer_outputs_with_hf_tokenizer


@pytest.fixture
def toy_tokenizer_from_files(test_dir):
    return RobertaTokenizer.from_files(
        vocab_path=test_dir / "toy-vocab.json", merges_path=test_dir / "toy-merges.txt"
    )


@pytest.fixture
def toy_tokenizer_from_tokenizer_json(test_dir):
    return RobertaTokenizer.from_tokenizer_json_file(test_dir / "toy-roberta.json")


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, "roberta-base", RobertaTokenizer
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_tokenizer_equals_hf_tokenizer(sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts,
        "roberta-base",
        RobertaTokenizer,
        from_hf_tokenizer=True,
        # Use a revision from before tokenizer.json.
        revision="2a6355b35ee37259407a8bb78148ad37ea87e39f",
    )


def test_roberta_tokenizer_roundtrip(sample_texts):
    tokenizer = RobertaTokenizer.from_hf_hub(name="roberta-base")
    pieces = tokenizer(sample_texts)
    decoded = tokenizer.decode(pieces.ids)

    assert decoded == sample_texts


def test_from_json_file(toy_tokenizer_from_tokenizer_json, short_sample_texts):
    encoding = toy_tokenizer_from_tokenizer_json(short_sample_texts)
    _check_toy_tokenizer(encoding)


def test_from_files(toy_tokenizer_from_files, short_sample_texts):
    encoding = toy_tokenizer_from_files(short_sample_texts)
    _check_toy_tokenizer(encoding)


def _check_toy_tokenizer(pieces):
    assert isinstance(pieces, PiecesWithIds)
    assert len(pieces.ids) == 2
    assert len(pieces.pieces) == 2

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
