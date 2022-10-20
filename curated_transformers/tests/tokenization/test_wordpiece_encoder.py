from functools import partial
import numpy.testing
from pathlib import Path
import pytest
import spacy
from thinc.api import Ragged, registry

from curated_transformers.tokenization.hf_loader import build_hf_piece_encoder_loader_v1
from curated_transformers.tokenization.wordpiece_encoder import build_wordpiece_encoder
from curated_transformers._compat import has_hf_transformers


@pytest.fixture
def toy_model_path(test_dir):
    return test_dir / "toy.wordpieces"


@pytest.fixture
def toy_encoder(toy_model_path):
    encoder = build_wordpiece_encoder()
    encoder.init = registry.model_loaders.get(
        "curated-transformers.WordpieceLoader.v1"
    )(path=toy_model_path)
    encoder.initialize()
    return encoder


def test_wordpiece_encoder_local_model(toy_encoder):
    _test_toy_encoder(toy_encoder)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_wordpiece_encoder_hf_model():
    nlp = spacy.blank("en")
    encoder = build_wordpiece_encoder()
    encoder.init = build_hf_piece_encoder_loader_v1(name="bert-base-cased")
    encoder.initialize()

    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")

    encoding = encoder.predict([doc1, doc2])

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    numpy.testing.assert_equal(encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    numpy.testing.assert_equal(
        encoding[0].dataXd, [101, 146, 1486, 170, 1873, 1114, 170, 16737, 119, 102]
    )

    numpy.testing.assert_equal(encoding[1].lengths, [1, 1, 1, 1, 1, 3, 1, 1, 1])
    numpy.testing.assert_equal(
        encoding[1].dataXd,
        [101, 3570, 1195, 1209, 3940, 185, 5926, 2744, 7329, 119, 102],
    )


def test_serialize(toy_encoder):
    encoder_bytes = toy_encoder.to_bytes()
    toy_encoder2 = build_wordpiece_encoder()
    toy_encoder2.from_bytes(encoder_bytes)
    assert (
        toy_encoder.attrs["wordpiece_processor"].to_list()
        == toy_encoder2.attrs["wordpiece_processor"].to_list()
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_serialize_hf_model():
    encoder = build_wordpiece_encoder()
    encoder.init = build_hf_piece_encoder_loader_v1(name="bert-base-cased")
    encoder.initialize()
    encoder_bytes = encoder.to_bytes()
    encoder2 = build_wordpiece_encoder()
    encoder2.from_bytes(encoder_bytes)
    assert (
        encoder.attrs["wordpiece_processor"].to_list()
        == encoder2.attrs["wordpiece_processor"].to_list()
    )


def _test_toy_encoder(encoder):
    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")

    encoding = encoder.predict([doc1, doc2])

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    numpy.testing.assert_equal(encoding[0].lengths, [1, 1, 1, 1, 3, 1, 1, 5, 1, 1])
    numpy.testing.assert_equal(
        encoding[0].dataXd,
        [2, 41, 818, 61, 67, 193, 88, 204, 61, 251, 909, 682, 102, 95, 17, 3],
    )

    numpy.testing.assert_equal(encoding[1].lengths, [1, 3, 1, 1, 2, 3, 3, 1, 1])
    numpy.testing.assert_equal(
        encoding[1].dataXd,
        [2, 824, 98, 189, 311, 417, 65, 155, 503, 99, 1, 416, 117, 88, 17, 3],
    )
