from cutlery import ByteBPEProcessor, WordPieceProcessor
from functools import partial
import numpy.testing
from pathlib import Path
import pytest
import spacy
from thinc.api import Ragged

from curated_transformers.tokenization.bbpe_encoder import build_byte_bpe_encoder
from curated_transformers.tokenization.hf_loader import build_hf_piece_encoder_loader
from curated_transformers._compat import has_hf_transformers


@pytest.fixture(scope="module")
def test_dir(request):
    return Path(request.fspath).parent


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_bbpe_encoder_hf_model():
    encoder = build_byte_bpe_encoder()
    encoder.init = build_hf_piece_encoder_loader(name="roberta-base")
    encoder.initialize()
    _check_roberta_base_encoder(encoder)


def test_serialize():
    processor = ByteBPEProcessor(
        {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "aa": 4, "b": 5}, [("a", "a")]
    )
    encoder = build_byte_bpe_encoder()
    encoder.attrs["byte_bpe_processor"] = processor
    encoder_bytes = encoder.to_bytes()
    encoder2 = build_byte_bpe_encoder()
    encoder2.from_bytes(encoder_bytes)

    nlp = spacy.blank("en")
    doc = nlp.make_doc("aab")
    encoding = encoder.predict([doc])
    numpy.testing.assert_equal(encoding[0].lengths, [1, 2, 1])
    numpy.testing.assert_equal(encoding[0].dataXd, [0, 4, 5, 2])


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_serialize_hf_model():
    encoder = build_byte_bpe_encoder()
    encoder.init = build_hf_piece_encoder_loader(name="roberta-base")
    encoder.initialize()
    encoder_bytes = encoder.to_bytes()
    encoder2 = build_byte_bpe_encoder()
    encoder2.from_bytes(encoder_bytes)
    _check_roberta_base_encoder(encoder)


def _check_roberta_base_encoder(encoder):
    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")

    encoding = encoder.predict([doc1, doc2])

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    numpy.testing.assert_equal(encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    numpy.testing.assert_equal(
        encoding[0].dataXd, [0, 100, 794, 10, 1816, 19, 10, 27608, 4, 2]
    )

    numpy.testing.assert_equal(encoding[1].lengths, [1, 1, 1, 1, 1, 2, 1, 1, 1])
    numpy.testing.assert_equal(
        encoding[1].dataXd, [0, 5625, 52, 40, 3529, 181, 48344, 5749, 4, 2]
    )
