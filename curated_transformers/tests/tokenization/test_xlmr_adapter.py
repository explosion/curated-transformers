from typing import List
from cutlery import SentencePieceProcessor
import numpy.testing
from pathlib import Path

import numpy.testing
import pytest
from thinc.api import NumpyOps, Ragged, chain, Model

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.tokenization.sentencepiece_encoder import (
    build_sentencepiece_encoder,
)
from curated_transformers.tokenization.sentencepiece_adapters import (
    build_xlmr_adapter,
)
from curated_transformers.tokenization.wordpiece_encoder import build_wordpiece_encoder
from curated_transformers.tokenization.hf_loader import build_hf_piece_encoder_loader_v1
from curated_transformers.models.output import TransformerModelOutput
from curated_transformers.models.remove_eos_bos import remove_bos_eos


@pytest.fixture
def toy_model(test_dir):
    return SentencePieceProcessor.from_file(str(test_dir / "toy.model"))


@pytest.fixture
def toy_encoder(toy_model):
    encoder = build_sentencepiece_encoder()
    encoder.attrs["sentencepiece_processor"] = toy_model
    return chain(encoder, build_xlmr_adapter())


@pytest.fixture
def empty_encoder():
    return chain(build_sentencepiece_encoder(), build_xlmr_adapter())


def _mock_transformer() -> Model[List[Ragged], TransformerModelOutput]:
    def forward(model: Model, X: List[Ragged], is_train: bool):
        return (
            TransformerModelOutput(outputs=[[x] for x in X], last_layer_only=True),
            lambda x: x,
        )

    return Model("mock_transformer", forward)


def test_sentencepiece_encoder(toy_encoder, sample_docs):
    encoding = toy_encoder.predict(sample_docs)
    _check_encoder(encoding)


def test_serialize(toy_encoder, empty_encoder, sample_docs):
    encoder_bytes = toy_encoder.to_bytes()
    encoder2 = empty_encoder
    encoder2.from_bytes(encoder_bytes)
    encoding = encoder2.predict(sample_docs)
    _check_encoder(encoding)


def _compare_model_hf_output(ops, Y, Y_hf):
    # Get inputs, removing BOS/EOS from every token.
    Y_hf = [e[1:-1] for e in Y_hf["input_ids"]]
    numpy.testing.assert_equal(ops.unflatten(Y.dataXd, Y.lengths), Y_hf)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_sentencepiece_encoder_against_hf(sample_docs):
    ops = NumpyOps()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    encoder = build_sentencepiece_encoder()
    encoder.init = build_hf_piece_encoder_loader_v1(name="xlm-roberta-base")
    encoder.initialize()
    model = chain(encoder, build_xlmr_adapter(), _mock_transformer(), remove_bos_eos())

    encoding = model.predict(sample_docs)
    hf_encoding = hf_tokenizer([token.text for token in sample_docs[0]])
    _compare_model_hf_output(ops, encoding.all_outputs[0][0], hf_encoding)

    hf_encoding = hf_tokenizer([token.text for token in sample_docs[1]])
    _compare_model_hf_output(ops, encoding.all_outputs[1][0], hf_encoding)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_wordpiece_encoder_against_hf(sample_docs):
    ops = NumpyOps()
    encoder = build_wordpiece_encoder()
    encoder.init = build_hf_piece_encoder_loader_v1(name="bert-base-cased")
    encoder.initialize()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    model = chain(encoder, _mock_transformer(), remove_bos_eos())

    encoding = model.predict(sample_docs)
    hf_encoding = hf_tokenizer([token.text for token in sample_docs[0]])
    _compare_model_hf_output(ops, encoding.all_outputs[0][0], hf_encoding)

    hf_encoding = hf_tokenizer([token.text for token in sample_docs[1]])
    _compare_model_hf_output(ops, encoding.all_outputs[1][0], hf_encoding)


def _check_encoder(encoding):
    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    numpy.testing.assert_equal(encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 6, 2, 1])
    numpy.testing.assert_equal(
        encoding[0].dataXd,
        [0, 9, 466, 11, 948, 42, 11, 171, 169, 111, 29, 21, 144, 8, 5, 2],
    )

    assert isinstance(encoding[1], Ragged)
    numpy.testing.assert_equal(encoding[1].lengths, [1, 2, 1, 1, 1, 4, 3, 2, 1])
    numpy.testing.assert_equal(
        encoding[1].dataXd,
        [0, 484, 547, 113, 172, 568, 63, 21, 46, 3, 85, 116, 28, 8, 5, 2],
    )
