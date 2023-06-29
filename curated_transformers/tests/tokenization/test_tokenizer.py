from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.tokenization import Tokenizer

from ..util import torch_assertclose
from .util import compare_tokenizer_outputs_with_hf_tokenizer


@dataclass
class _TestModel:
    model: str
    pad_token: Optional[str] = None


_MODELS = [
    _TestModel("bert-base-cased"),
    _TestModel("camembert-base"),
    _TestModel("EleutherAI/gpt-neox-20b", pad_token="[PAD]"),
    _TestModel("roberta-base"),
    _TestModel("tiiuae/falcon-7b", pad_token="<|endoftext|>"),
    _TestModel("xlm-roberta-base"),
]

_FRENCH_MODELS = [_TestModel("camembert-base")]


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


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_file(sample_texts):
    model_path = Path(__file__).parent / "toy-roberta" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(model_path)
    hf_tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
        str(model_path.parent)
    )

    pieces = tokenizer(sample_texts)
    hf_pieces = hf_tokenizer(sample_texts, padding=True, return_tensors="pt")
    torch_assertclose(
        pieces.padded_tensor(padding_id=hf_tokenizer.pad_token_id),
        hf_pieces["input_ids"].int(),
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_json(sample_texts):
    model_path = Path(__file__).parent / "toy-roberta" / "tokenizer.json"
    with open(model_path, encoding="utf-8") as f:
        tokenizer = Tokenizer.from_json(f.read())
    hf_tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
        str(model_path.parent)
    )

    pieces = tokenizer(sample_texts)
    hf_pieces = hf_tokenizer(sample_texts, padding=True, return_tensors="pt")
    torch_assertclose(
        pieces.padded_tensor(padding_id=hf_tokenizer.pad_token_id),
        hf_pieces["input_ids"].int(),
    )
