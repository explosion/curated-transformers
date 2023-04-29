import pytest
from cutlery import SentencePieceProcessor
from thinc.api import CupyOps, NumpyOps, Ragged, registry
from thinc.compat import has_cupy

from curated_transformers.models.architectures import (
    build_albert_transformer_model_v1,
    build_bert_transformer_model_v1,
    build_roberta_transformer_model_v1,
    build_xlmr_transformer_model_v1,
)
from curated_transformers.models.hf_loader import build_hf_transformer_encoder_loader_v1
from curated_transformers.models.output import TransformerModelOutput
from curated_transformers.models.with_strided_spans import build_with_strided_spans_v1
from curated_transformers.tokenization import (
    build_bert_wordpiece_encoder_v1,
    build_byte_bpe_encoder_v1,
    build_hf_piece_encoder_loader_v1,
    build_sentencepiece_encoder_v1,
    build_xlmr_sentencepiece_encoder_v1,
)
from curated_transformers._compat import (
    has_hf_transformers,
    has_huggingface_hub,
)


OPS = [NumpyOps()]
if has_cupy:
    OPS.append(CupyOps())


@pytest.fixture
def toy_model(test_dir):
    return SentencePieceProcessor.from_file(
        (str(test_dir / ".." / "tokenization" / "toy.model"))
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("stride,window", [(2, 4), (96, 128)])
@pytest.mark.parametrize("hf_model", [("xlm-roberta-base", 768, 250002)])
def test_xlmr_model(sample_docs, stride, window, hf_model):
    hf_model_name, hidden_width, vocab_size = hf_model
    with_spans = build_with_strided_spans_v1(stride=stride, window=window)
    model = build_xlmr_transformer_model_v1(
        piece_encoder=build_xlmr_sentencepiece_encoder_v1(),
        with_spans=with_spans,
        vocab_size=vocab_size,
        hidden_width=hidden_width,
    )
    model.get_ref("transformer").init = build_hf_transformer_encoder_loader_v1(
        name=hf_model_name
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name=hf_model_name
    )
    model.initialize(X=sample_docs)
    Y, backprop = model(sample_docs, is_train=False)
    assert isinstance(Y, TransformerModelOutput)
    num_ouputs = Y.num_outputs
    Y = Y.last_hidden_layer_states
    assert len(Y) == 2
    model.ops.xp.testing.assert_array_equal(Y[0].lengths, [1, 1, 1, 1, 1, 1, 2, 2])
    assert Y[0].dataXd.shape == (10, hidden_width)
    model.ops.xp.testing.assert_array_equal(Y[1].lengths, [1, 1, 1, 1, 2, 1, 2])
    assert Y[1].dataXd.shape == (9, hidden_width)

    # Backprop zeros to verify that backprop doesn't fail.
    dY = [
        [
            Ragged(
                model.ops.alloc2f(10, 768),
                lengths=model.ops.asarray1i([1, 1, 1, 1, 1, 1, 2, 2]),
            )
            for _ in range(num_ouputs)
        ],
        [
            Ragged(
                model.ops.alloc2f(9, 768),
                lengths=model.ops.asarray1i([1, 1, 1, 1, 2, 1, 2]),
            )
            for _ in range(num_ouputs)
        ],
    ]
    assert backprop(dY) == []


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("stride,window", [(2, 4), (96, 128)])
@pytest.mark.parametrize("hf_model", [("xlm-roberta-base", 768, 250002)])
def test_input_with_spaces(sample_docs_with_spaces, stride, window, hf_model):
    hidden_width = 768
    with_spans = build_with_strided_spans_v1(stride=stride, window=window)
    model = build_xlmr_transformer_model_v1(
        piece_encoder=build_xlmr_sentencepiece_encoder_v1(),
        with_spans=with_spans,
        vocab_size=250005,
        hidden_width=hidden_width,
        num_hidden_layers=1,
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.initialize(X=sample_docs_with_spaces)
    Y, backprop = model(sample_docs_with_spaces, is_train=False)
    assert isinstance(Y, TransformerModelOutput)
    num_ouputs = Y.num_outputs
    Y = Y.last_hidden_layer_states
    assert len(Y) == 2
    model.ops.xp.testing.assert_array_equal(Y[0].lengths, [1, 1, 1, 1, 1, 1, 1, 2, 2])
    assert Y[0].dataXd.shape == (11, hidden_width)
    model.ops.xp.testing.assert_array_equal(
        Y[1].lengths, [1, 1, 1, 1, 1, 1, 2, 1, 2, 1]
    )
    assert Y[1].dataXd.shape == (12, hidden_width)

    # Backprop zeros to verify that backprop doesn't fail.
    dY = [
        [
            Ragged(
                model.ops.alloc2f(11, 768),
                lengths=model.ops.asarray1i([1, 1, 1, 1, 1, 1, 1, 2, 2]),
            )
            for _ in range(num_ouputs)
        ],
        [
            Ragged(
                model.ops.alloc2f(11, 768),
                lengths=model.ops.asarray1i([1, 1, 1, 1, 1, 2, 1, 2, 1]),
            )
            for _ in range(num_ouputs)
        ],
    ]
    assert backprop(dY) == []


@pytest.mark.slow
@pytest.mark.skipif(not has_huggingface_hub, reason="requires huggingface hub")
@pytest.mark.parametrize(
    "test_config",
    [
        (
            "albert-base-v2",
            build_albert_transformer_model_v1,
            build_sentencepiece_encoder_v1(),
            30000,
        ),
        (
            "bert-base-cased",
            build_bert_transformer_model_v1,
            build_bert_wordpiece_encoder_v1(),
            28996,
        ),
        (
            "roberta-base",
            build_roberta_transformer_model_v1,
            build_byte_bpe_encoder_v1(),
            50265,
        ),
    ],
)
def test_pytorch_checkpoint_loader(test_config):
    from huggingface_hub import hf_hub_download

    model_name, model_factory, piece_encoder, vocab_size = test_config

    checkpoint_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    with_spans = build_with_strided_spans_v1(stride=96, window=128)
    model = model_factory(
        piece_encoder=piece_encoder, vocab_size=vocab_size, with_spans=with_spans
    )
    model.get_ref("transformer").init = registry.model_loaders.get(
        "curated-transformers.PyTorchCheckpointLoader.v1"
    )(path=checkpoint_path)
    model.initialize()
