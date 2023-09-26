from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from curated_transformers.tokenizers import Tokenizer
from curated_transformers.tokenizers.chunks import InputChunks, TextChunk

from ..compat import has_hf_transformers, transformers
from ..utils import torch_assertclose
from .util import compare_tokenizer_outputs_with_hf_tokenizer


@dataclass
class _TestModel:
    model: str
    pad_token: Optional[str] = None
    eos_piece: Optional[str] = None


_MODELS = [
    _TestModel("bert-base-cased"),
    _TestModel("camembert-base"),
    _TestModel("EleutherAI/gpt-neox-20b", pad_token="[PAD]", eos_piece="<|endoftext|>"),
    _TestModel("ausboss/llama-30b-supercot", pad_token="</s>", eos_piece="</s>"),
    _TestModel("roberta-base"),
    _TestModel(
        "tiiuae/falcon-7b", pad_token="<|endoftext|>", eos_piece="<|endoftext|>"
    ),
    _TestModel("xlm-roberta-base"),
]

_FRENCH_MODELS = [_TestModel("camembert-base")]


@pytest.fixture
def toy_tokenizer_path():
    return Path(__file__).parent / "toy-roberta"


@pytest.fixture
def toy_tokenizer(toy_tokenizer_path):
    return Tokenizer.from_dir(toy_tokenizer_path)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("model", _MODELS)
def test_against_hf_tokenizers(model, sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts, model.model, Tokenizer, pad_token=model.pad_token
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("model", _MODELS)
def test_against_hf_tokenizers_short(model, short_sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        short_sample_texts, model.model, Tokenizer, pad_token=model.pad_token
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("model", _FRENCH_MODELS)
def test_against_hf_tokenizers_french(model, french_sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(
        french_sample_texts, model.model, Tokenizer, pad_token=model.pad_token
    )


@pytest.mark.parametrize("model", _MODELS)
def test_special_pieces(model):
    tokenizer = Tokenizer.from_hf_hub(name=model.model)
    assert tokenizer.eos_piece == model.eos_piece


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_dir(toy_tokenizer, toy_tokenizer_path, sample_texts):
    hf_tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
        str(toy_tokenizer_path)
    )

    pieces = toy_tokenizer(sample_texts)
    hf_pieces = hf_tokenizer(sample_texts, padding=True, return_tensors="pt")
    torch_assertclose(
        pieces.padded_tensor(padding_id=hf_tokenizer.pad_token_id),
        hf_pieces["input_ids"].int(),
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_json(toy_tokenizer_path, sample_texts):
    with open(toy_tokenizer_path / "tokenizer.json", encoding="utf-8") as f:
        tokenizer = Tokenizer.from_json(f.read())
    hf_tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
        str(toy_tokenizer_path)
    )

    pieces = tokenizer(sample_texts)
    hf_pieces = hf_tokenizer(sample_texts, padding=True, return_tensors="pt")
    torch_assertclose(
        pieces.padded_tensor(padding_id=hf_tokenizer.pad_token_id),
        hf_pieces["input_ids"].int(),
    )


def test_invalid_string_input(toy_tokenizer):
    with pytest.raises(ValueError, match=r"Non-string.*float, int"):
        toy_tokenizer(["hello", 42, 3.14159])


def test_invalid_chunk_input(toy_tokenizer):
    with pytest.raises(ValueError, match=r"Non-chunk.*float, int"):
        toy_tokenizer([InputChunks([TextChunk("hello")]), 42, 3.14159])
