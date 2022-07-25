from cysp.sp import SentencePieceProcessor
import numpy.testing
from pathlib import Path
import pytest
import spacy
from thinc.api import Ragged

from spacy_experimental.transformers.tokenization.sentencepiece_encoder import (
    build_sentencepiece_encoder,
)


@pytest.fixture(scope="module")
def test_dir(request):
    return Path(request.fspath).parent


@pytest.fixture
def toy_model(test_dir):
    return SentencePieceProcessor.from_file(str(test_dir / "toy.model"))


@pytest.fixture
def toy_encoder(toy_model):
    encoder = build_sentencepiece_encoder()
    encoder.attrs["sentencepiece_processor"] = toy_model
    return encoder


def test_sentencepiece_encoder(toy_encoder):
    _test_encoder(toy_encoder)


def test_serialize(toy_encoder):
    encoder_bytes = toy_encoder.to_bytes()
    encoder2 = build_sentencepiece_encoder()
    encoder2.from_bytes(encoder_bytes)
    _test_encoder(encoder2)


def _test_encoder(encoder):
    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")

    encoding = encoder.predict([doc1, doc2])

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    numpy.testing.assert_equal(encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 6, 2, 1])
    numpy.testing.assert_equal(
        encoding[0].dataXd,
        [1, 8, 465, 10, 947, 41, 10, 170, 168, 110, 28, 20, 143, 7, 4, 2],
    )

    assert isinstance(encoding[1], Ragged)
    numpy.testing.assert_equal(encoding[1].lengths, [1, 2, 1, 1, 1, 4, 3, 2, 1])
    numpy.testing.assert_equal(
        encoding[1].dataXd,
        [1, 483, 546, 112, 171, 567, 62, 20, 45, 0, 84, 115, 27, 7, 4, 2],
    )
