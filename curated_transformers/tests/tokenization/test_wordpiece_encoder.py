import pytest
import spacy
from tempfile import TemporaryDirectory
from thinc.api import Ragged, get_current_ops

from curated_transformers.tokenization.hf_loader import build_hf_piece_encoder_loader_v1
from curated_transformers.tokenization.wordpiece_encoder import (
    build_bert_wordpiece_encoder_v1,
    build_wordpiece_encoder_v1,
    _bert_preprocess,
    build_wordpiece_encoder_loader_v1,
)
from curated_transformers._compat import has_hf_transformers, transformers


def test_wordpiece_encoder_local_model(wordpiece_toy_encoder, sample_docs):
    encoding = wordpiece_toy_encoder.predict(sample_docs)
    _check_toy_encoder(encoding)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_wordpiece_encoder_hf_model(sample_docs):
    ops = get_current_ops()
    encoder = build_wordpiece_encoder_v1()
    encoder.init = build_hf_piece_encoder_loader_v1(name="bert-base-cased")
    encoder.initialize()

    encoding = encoder.predict(sample_docs)

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops.xp.testing.assert_array_equal(
        encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd, [101, 146, 1486, 170, 1873, 1114, 170, 16737, 119, 102]
    )

    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 1, 1, 1, 1, 3, 1, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd,
        [101, 3570, 1195, 1209, 3940, 185, 5926, 2744, 7329, 119, 102],
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_wordpiece_encoder_hf_model_uncased(sample_docs):
    ops = get_current_ops()
    encoder = build_wordpiece_encoder_v1()
    encoder.init = build_hf_piece_encoder_loader_v1(name="bert-base-uncased")
    encoder.initialize()

    encoding = encoder.predict(sample_docs)

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops.xp.testing.assert_array_equal(
        encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd, [101, 1045, 2387, 1037, 2611, 2007, 1037, 12772, 1012, 102]
    )

    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 1, 1, 1, 1, 1, 1, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd, [101, 2651, 2057, 2097, 4521, 26202, 4605, 1012, 102]
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_wordpiece_encoder_hf_model_german():
    ops = get_current_ops()
    encoder = build_bert_wordpiece_encoder_v1()
    encoder.init = build_hf_piece_encoder_loader_v1(name="bert-base-german-cased")
    encoder.initialize()

    nlp = spacy.blank("de")
    sample_docs = [
        nlp.make_doc("Wir sehen ein AWO-Mitarbeiter."),
        nlp.make_doc("Die Mw.-St. beträgt 19 Prozent."),
    ]

    encoding = encoder.predict(sample_docs)

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops.xp.testing.assert_array_equal(encoding[0].lengths, [1, 1, 1, 1, 5, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd, [3, 655, 2265, 39, 32, 26939, 26962, 26935, 2153, 26914, 4]
    )

    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 1, 5, 1, 1, 1, 1, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd,
        [3, 125, 56, 26915, 26914, 26935, 130, 26914, 4490, 141, 1028, 26914, 4],
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_wordpiece_encoder_loader(sample_docs):
    ops = get_current_ops()
    encoder = build_wordpiece_encoder_v1()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    with TemporaryDirectory() as d:
        vocab_path = hf_tokenizer.save_vocabulary(d)[0]
        encoder.init = build_wordpiece_encoder_loader_v1(path=vocab_path)
        encoder.initialize()

    encoding = encoder.predict(sample_docs)

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops.xp.testing.assert_array_equal(
        encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd, [101, 146, 1486, 170, 1873, 1114, 170, 16737, 119, 102]
    )

    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 1, 1, 1, 1, 3, 1, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd,
        [101, 3570, 1195, 1209, 3940, 185, 5926, 2744, 7329, 119, 102],
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_wordpiece_encoder_loader_uncased(sample_docs):
    ops = get_current_ops()
    encoder = build_wordpiece_encoder_v1()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    with TemporaryDirectory() as d:
        vocab_path = hf_tokenizer.save_vocabulary(d)[0]
        encoder.init = build_wordpiece_encoder_loader_v1(
            path=vocab_path, lowercase=True, strip_accents=True
        )
        encoder.initialize()

    encoding = encoder.predict(sample_docs)

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops.xp.testing.assert_array_equal(
        encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd, [101, 1045, 2387, 1037, 2611, 2007, 1037, 12772, 1012, 102]
    )

    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 1, 1, 1, 1, 1, 1, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd, [101, 2651, 2057, 2097, 4521, 26202, 4605, 1012, 102]
    )


def test_serialize(wordpiece_toy_encoder):
    encoder_bytes = wordpiece_toy_encoder.to_bytes()
    toy_encoder2 = build_wordpiece_encoder_v1()
    toy_encoder2.from_bytes(encoder_bytes)
    assert (
        wordpiece_toy_encoder.attrs["wordpiece_processor"].to_list()
        == toy_encoder2.attrs["wordpiece_processor"].to_list()
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_serialize_hf_model():
    encoder = build_wordpiece_encoder_v1()
    encoder.init = build_hf_piece_encoder_loader_v1(name="bert-base-cased")
    encoder.initialize()
    encoder_bytes = encoder.to_bytes()
    encoder2 = build_wordpiece_encoder_v1()
    encoder2.from_bytes(encoder_bytes)
    assert (
        encoder.attrs["wordpiece_processor"].to_list()
        == encoder2.attrs["wordpiece_processor"].to_list()
    )


def test_bert_preprocess():
    assert _bert_preprocess("AWO-Mitarbeiter") == ["AWO", "-", "Mitarbeiter"]
    assert _bert_preprocess("-Mitarbeiter") == ["-", "Mitarbeiter"]
    assert _bert_preprocess("AWO-") == ["AWO", "-"]
    assert _bert_preprocess("-") == ["-"]
    assert _bert_preprocess("") == []
    assert _bert_preprocess("Mw.-St.") == ["Mw", ".", "-", "St", "."]


def _check_toy_encoder(encoding):
    ops = get_current_ops()
    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops.xp.testing.assert_array_equal(
        encoding[0].lengths, [1, 1, 1, 1, 3, 1, 1, 5, 1, 1]
    )
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd,
        [2, 41, 818, 61, 67, 193, 88, 204, 61, 251, 909, 682, 102, 95, 17, 3],
    )

    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 3, 1, 1, 2, 3, 3, 1, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd,
        [2, 824, 98, 189, 311, 417, 65, 155, 503, 99, 1, 416, 117, 88, 17, 3],
    )
