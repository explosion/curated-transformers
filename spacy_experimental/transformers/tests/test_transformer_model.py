from pathlib import Path

import numpy
import pytest
import spacy
from cysp import SentencePieceProcessor
from thinc.api import CupyOps, NumpyOps, Ragged
from thinc.compat import has_cupy

from spacy_experimental.transformers.models.with_strided_spans import build_with_strided_spans
from spacy_experimental.transformers.models.transformer_model import (
    build_xlmr_transformer_model_v1,
)

OPS = [NumpyOps()]
if has_cupy:
    OPS.append(CupyOps())


@pytest.fixture(scope="module")
def test_dir(request):
    return Path(request.fspath).parent


@pytest.fixture
def toy_model(test_dir):
    return SentencePieceProcessor.from_file(
        (str(test_dir / ".." / "tokenization" / "tests" / "toy.model"))
    )


@pytest.fixture
def example_docs():
    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")
    return [doc1, doc2]


@pytest.mark.parametrize("stride,window", [(2, 4), (96, 128)])
def test_xlmr_model(example_docs, toy_model, stride, window):
    with_spans = build_with_strided_spans(stride=stride, window=window)
    model = build_xlmr_transformer_model_v1(with_spans=with_spans)
    piece_encoder = model.get_ref("piece_encoder")
    piece_encoder.attrs["sentencepiece_processor"] = toy_model
    model.initialize(X=example_docs)
    Y, backprop = model(example_docs, is_train=False)
    assert isinstance(Y, list)
    assert len(Y) == 2
    numpy.testing.assert_equal(Y[0].lengths, [1, 1, 1, 1, 1, 1, 1, 6, 2, 1])
    assert Y[0].dataXd.shape == (16, 768)
    numpy.testing.assert_equal(Y[1].lengths, [1, 2, 1, 1, 1, 4, 3, 2, 1])
    assert Y[1].dataXd.shape == (16, 768)

    # Backprop zeros to verify that backprop doesn't fail.
    ops = NumpyOps()
    dY = [
        Ragged(
            ops.alloc2f(16, 768), lengths=ops.asarray1i([1, 1, 1, 1, 1, 1, 1, 6, 2, 1])
        ),
        Ragged(
            ops.alloc2f(16, 768), lengths=ops.asarray1i([1, 2, 1, 1, 1, 4, 3, 2, 1])
        ),
    ]
    assert backprop(dY) == []
