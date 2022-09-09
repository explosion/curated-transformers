from cysp import SentencePieceProcessor
import numpy.testing
from pathlib import Path
import pytest
import spacy
from thinc.api import NumpyOps, Ragged, chain

from spacy_experimental.transformers._compat import has_hf_transformers, transformers
from spacy_experimental.transformers.tokenization.sentencepiece_encoder import build_sentencepiece_encoder
from spacy_experimental.transformers.tokenization.sentencepiece_adapters import build_xlmr_adapter


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
    return chain(encoder, build_xlmr_adapter())


@pytest.fixture
def empty_encoder():
    return chain(build_sentencepiece_encoder(), build_xlmr_adapter())


def test_sentencepiece_encoder(toy_encoder):
    _test_encoder(toy_encoder)


def test_serialize(toy_encoder, empty_encoder):
    encoder_bytes = toy_encoder.to_bytes()
    encoder2 = empty_encoder
    encoder2.from_bytes(encoder_bytes)
    _test_encoder(encoder2)


def _compare_model_hf_output(ops, Y, Y_hf):
    # Get inputs, removing BOS/EOS from every token.
    Y_hf = [e[1:-1] for e in Y_hf["input_ids"]]

    # Remove BOS/EOS
    Y = Y[1:-1]

    numpy.testing.assert_equal(ops.unflatten(Y.dataXd, Y.lengths), Y_hf)


@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_encoder_against_hf():
    ops = NumpyOps()

    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    spp = SentencePieceProcessor.from_file(hf_tokenizer.vocab_file)
    encoder = build_sentencepiece_encoder()
    encoder.attrs["sentencepiece_processor"] = spp
    model = chain(encoder, build_xlmr_adapter())

    encoding = model.predict([doc1, doc2])
    hf_encoding = hf_tokenizer([token.text for token in doc1])
    _compare_model_hf_output(ops, encoding[0], hf_encoding)

    hf_encoding = hf_tokenizer([token.text for token in doc2])
    _compare_model_hf_output(ops, encoding[1], hf_encoding)


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
        [0, 9, 466, 11, 948, 42, 11, 171, 169, 111, 29, 21, 144, 8, 5, 2],
    )

    assert isinstance(encoding[1], Ragged)
    numpy.testing.assert_equal(encoding[1].lengths, [1, 2, 1, 1, 1, 4, 3, 2, 1])
    numpy.testing.assert_equal(
        encoding[1].dataXd,
        [0, 484, 547, 113, 172, 568, 63, 21, 46, 3, 85, 116, 28, 8, 5, 2],
    )
