import pytest
from thinc.api import Ragged, get_current_ops
import spacy

from curated_transformers.tokenization.char_encoder import (
    build_char_encoder_loader_v1,
    build_char_encoder_v1,
)
from curated_transformers.tokenization.hf_loader import build_hf_piece_encoder_loader_v1
from curated_transformers.tokenization.wordpiece_encoder import (
    build_wordpiece_encoder_v1,
)
from curated_transformers.util import registry
from curated_transformers._compat import has_fugashi, has_hf_transformers, has_sudachi


def test_char_encoder(test_dir):
    ops = get_current_ops()
    encoder = build_char_encoder_v1()
    encoder.init = build_char_encoder_loader_v1(path=test_dir / "toy-chars.txt")
    encoder.initialize()

    nlp = spacy.blank("nl")
    sample_docs = [nlp.make_doc("Zeeën van tijd."), nlp.make_doc("Geïnd geld")]

    encoding = encoder.predict(sample_docs)

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops.xp.testing.assert_array_equal(encoding[0].lengths, [1, 5, 3, 4, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd, [2, 56, 9, 9, 57, 18, 26, 5, 18, 24, 13, 14, 8, 1, 3]
    )

    assert isinstance(encoding[1], Ragged)
    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 5, 4, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd, [2, 37, 9, 1, 18, 8, 11, 9, 16, 8, 3]
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_fugashi, reason="requires fugashi")
@pytest.mark.skipif(not has_sudachi, reason="requires SudachiPy and SudachiDict-core")
def test_char_encoder_hf_model():
    ops = get_current_ops()
    encoder = build_char_encoder_v1()
    encoder.init = registry.model_loaders.get(
        "curated-transformers.HFPieceEncoderLoader.v1"
    )(name="cl-tohoku/bert-base-japanese-char-v2")
    encoder.initialize()

    nlp = spacy.blank("ja")
    sample_docs = [nlp.make_doc("日本語だよ"), nlp.make_doc("吾輩は猫である。")]

    encoding = encoder.predict(sample_docs)

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops.xp.testing.assert_array_equal(encoding[0].lengths, [1, 2, 1, 1, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd, [2, 2719, 2828, 4923, 882, 922, 3]
    )

    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 2, 1, 1, 1, 2, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd, [2, 1583, 5159, 897, 3574, 889, 852, 925, 829, 3]
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_fugashi, reason="requires fugashi")
@pytest.mark.skipif(not has_sudachi, reason="requires SudachiPy and SudachiDict-core")
def test_hf_loader_rejects_incorrect_encoder():
    encoder = build_wordpiece_encoder_v1()
    encoder.init = build_hf_piece_encoder_loader_v1(
        name="cl-tohoku/bert-base-japanese-char-v2"
    )
    with pytest.raises(ValueError, match="incompatible piece encoder"):
        encoder.initialize()


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_fugashi, reason="requires fugashi")
@pytest.mark.skipif(not has_sudachi, reason="requires SudachiPy and SudachiDict-core")
def test_hf_loader_rejects_incorrect_model():
    encoder = build_char_encoder_v1()
    encoder.init = build_hf_piece_encoder_loader_v1(
        name="cl-tohoku/bert-base-japanese-v2"
    )
    with pytest.raises(ValueError, match="only support character subword"):
        encoder.initialize()


@pytest.mark.slow
def test_loader_rejects_incorrect_encoder(test_dir):
    encoder = build_wordpiece_encoder_v1()
    encoder.init = build_char_encoder_loader_v1(path=test_dir / "toy-chars.txt")
    with pytest.raises(ValueError, match="incompatible model"):
        encoder.initialize()
