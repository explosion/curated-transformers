import numpy.testing
import pytest
from thinc.api import Ragged, registry

from curated_transformers.tokenization.bbpe_encoder import build_byte_bpe_encoder
from curated_transformers.tokenization.hf_loader import build_hf_piece_encoder_loader_v1
from curated_transformers._compat import has_hf_transformers


@pytest.fixture
def toy_encoder(test_dir):
    encoder = build_byte_bpe_encoder()
    encoder.init = registry.model_loaders.get("curated-transformers.ByteBPELoader.v1")(
        vocab_path=test_dir / "toy-vocab.json", merges_path=test_dir / "toy-merges.txt"
    )
    encoder.initialize()
    return encoder


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_bbpe_encoder_hf_model(sample_docs):
    encoder = build_byte_bpe_encoder()
    encoder.init = build_hf_piece_encoder_loader_v1(name="roberta-base")
    encoder.initialize()
    encoding = encoder.predict(sample_docs)
    _check_roberta_base_encoder(encoding)


def test_bbpe_encoder(toy_encoder, sample_docs):
    encoding = toy_encoder.predict(sample_docs)
    _check_toy_encoder(encoding)


def test_serialize(toy_encoder):
    encoder_bytes = toy_encoder.to_bytes()
    toy_encoder2 = build_byte_bpe_encoder()
    toy_encoder2.from_bytes(encoder_bytes)
    assert (
        toy_encoder.attrs["byte_bpe_processor"].vocab
        == toy_encoder2.attrs["byte_bpe_processor"].vocab
    )
    assert (
        toy_encoder.attrs["byte_bpe_processor"].merges
        == toy_encoder2.attrs["byte_bpe_processor"].merges
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_serialize_hf_model(sample_docs):
    encoder = build_byte_bpe_encoder()
    encoder.init = build_hf_piece_encoder_loader_v1(name="roberta-base")
    encoder.initialize()
    encoder_bytes = encoder.to_bytes()
    encoder2 = build_byte_bpe_encoder()
    encoder2.from_bytes(encoder_bytes)
    encoding = encoder2.predict(sample_docs)
    _check_roberta_base_encoder(encoding)


def _check_toy_encoder(encoding):
    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    numpy.testing.assert_equal(encoding[0].lengths, [1, 1, 1, 1, 3, 1, 1, 6, 1, 1])
    numpy.testing.assert_equal(
        encoding[0].dataXd,
        [0, 44, 997, 262, 305, 334, 79, 342, 262, 388, 79, 302, 70, 472, 72, 17, 2],
    )

    numpy.testing.assert_equal(encoding[1].lengths, [1, 3, 1, 1, 2, 4, 3, 1, 1])
    numpy.testing.assert_equal(
        encoding[1].dataXd,
        [0, 55, 841, 321, 362, 579, 324, 294, 291, 494, 131, 106, 270, 307, 79, 17, 2],
    )


def _check_roberta_base_encoder(encoding):
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
