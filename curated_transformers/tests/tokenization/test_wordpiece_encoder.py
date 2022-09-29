from cutlery import SentencePieceProcessor, WordPieceProcessor
from functools import partial
import numpy.testing
from pathlib import Path
import pytest
import spacy
from thinc.api import Ragged

from curated_transformers.tokenization.wordpiece_encoder import build_hf_wordpiece_encoder_loader
from curated_transformers.tokenization.wordpiece_encoder import build_wordpiece_encoder
from curated_transformers._compat import has_hf_transformers


@pytest.fixture(scope="module")
def test_dir(request):
    return Path(request.fspath).parent


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_wordpiece_encoder_hf_model():
    nlp = spacy.blank("en")
    encoder = build_wordpiece_encoder(
        init=build_hf_wordpiece_encoder_loader(name="bert-base-cased")
    )
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


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_sentencepiece_encoder_unsupported_hf_model():
    encoder = build_wordpiece_encoder(
        init=build_hf_wordpiece_encoder_loader(name="roberta-base")
    )
    with pytest.raises(ValueError, match=r"not supported"):
        encoder.initialize()


def test_serialize():
    processor = WordPieceProcessor(["this", "##is", "a", "##test"])
    encoder = build_wordpiece_encoder()
    encoder.attrs["wordpiece_processor"] = processor
    encoder_bytes = encoder.to_bytes()
    encoder2 = build_wordpiece_encoder()
    encoder2.from_bytes(encoder_bytes)
    assert (
        encoder.attrs["wordpiece_processor"].to_list()
        == encoder2.attrs["wordpiece_processor"].to_list()
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_serialize_hf_model():
    encoder = build_wordpiece_encoder(
        init=build_hf_wordpiece_encoder_loader(name="bert-base-cased")
    )
    encoder.initialize()
    encoder_bytes = encoder.to_bytes()
    encoder2 = build_wordpiece_encoder()
    encoder2.from_bytes(encoder_bytes)
    assert (
        encoder.attrs["wordpiece_processor"].to_list()
        == encoder2.attrs["wordpiece_processor"].to_list()
    )
