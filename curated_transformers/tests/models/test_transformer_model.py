from pathlib import Path
import numpy
import pytest
import spacy
from cutlery import SentencePieceProcessor
from thinc.api import CupyOps, NumpyOps, Ragged, registry
from thinc.compat import has_cupy

from curated_transformers.models.hf_loader import build_hf_encoder_loader_v1
from curated_transformers.models.output import TransformerModelOutput
from curated_transformers.models.with_strided_spans import build_with_strided_spans_v1
from curated_transformers.models.transformer_model import (
    build_albert_transformer_model_v1,
    build_bert_transformer_model_v1,
    build_roberta_transformer_model_v1,
    build_xlmr_transformer_model_v1,
)
from curated_transformers.tokenization.hf_loader import build_hf_piece_encoder_loader_v1
from curated_transformers._compat import (
    has_hf_transformers,
    has_huggingface_hub,
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
        (str(test_dir / ".." / "tokenization" / "toy.model"))
    )


@pytest.fixture
def example_docs():
    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")
    return [doc1, doc2]


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("stride,window", [(2, 4), (96, 128)])
@pytest.mark.parametrize("hf_model", [("xlm-roberta-base", 768, 250002)])
def test_xlmr_model(example_docs, toy_model, stride, window, hf_model):
    hf_model_name, hidden_size, vocab_size = hf_model
    with_spans = build_with_strided_spans_v1(stride=stride, window=window)
    model = build_xlmr_transformer_model_v1(
        with_spans=with_spans,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
    model.get_ref("transformer").init = build_hf_encoder_loader_v1(name=hf_model_name)
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name=hf_model_name
    )
    model.initialize(X=example_docs)
    Y, backprop = model(example_docs, is_train=False)
    assert isinstance(Y, TransformerModelOutput)
    num_ouputs = Y.num_outputs
    Y = Y.last_hidden_states
    assert len(Y) == 2
    numpy.testing.assert_equal(Y[0].lengths, [1, 1, 1, 1, 1, 1, 2, 2])
    assert Y[0].dataXd.shape == (10, hidden_size)
    numpy.testing.assert_equal(Y[1].lengths, [1, 1, 1, 1, 2, 1, 2])
    assert Y[1].dataXd.shape == (9, hidden_size)

    # Backprop zeros to verify that backprop doesn't fail.
    ops = NumpyOps()
    dY = [
        [
            Ragged(
                ops.alloc2f(10, 768), lengths=ops.asarray1i([1, 1, 1, 1, 1, 1, 2, 2])
            )
            for _ in range(num_ouputs)
        ],
        [
            Ragged(ops.alloc2f(9, 768), lengths=ops.asarray1i([1, 1, 1, 1, 2, 1, 2]))
            for _ in range(num_ouputs)
        ],
    ]
    assert backprop(dY) == []


@pytest.mark.slow
@pytest.mark.skipif(not has_huggingface_hub, reason="requires 🤗 hub")
@pytest.mark.parametrize(
    "test_config",
    [
        ("albert-base-v2", build_albert_transformer_model_v1, 30000),
        (
            "bert-base-cased",
            build_bert_transformer_model_v1,
            28996,
        ),
        ("roberta-base", build_roberta_transformer_model_v1, 50265),
    ],
)
def test_pytorch_checkpoint_loader(test_config):
    from huggingface_hub import hf_hub_download

    model_name, model_factory, vocab_size = test_config

    checkpoint_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    with_spans = build_with_strided_spans_v1(stride=96, window=128)
    model = model_factory(vocab_size=vocab_size, with_spans=with_spans)
    model.get_ref("transformer").init = registry.model_loaders.get(
        "curated-transformers.PyTorchCheckpointLoader.v1"
    )(path=checkpoint_path)
    model.initialize()
