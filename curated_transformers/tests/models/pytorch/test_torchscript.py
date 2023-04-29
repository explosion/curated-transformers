import copy
import pytest
import torch.jit

from curated_transformers.models.with_strided_spans import (
    build_with_strided_spans_v1,
    with_strided_spans,
)
from curated_transformers.tokenization.wordpiece_encoder import (
    build_wordpiece_encoder_v1,
)
from curated_transformers.models.architectures import (
    build_albert_transformer_model_v1,
    build_bert_transformer_model_v1,
    build_camembert_transformer_model_v1,
    build_roberta_transformer_model_v1,
    build_xlmr_transformer_model_v1,
)

MODEL_CONSTRUCTORS = [
    build_albert_transformer_model_v1,
    build_bert_transformer_model_v1,
    build_camembert_transformer_model_v1,
    build_roberta_transformer_model_v1,
    build_xlmr_transformer_model_v1,
]


@pytest.mark.slow
@pytest.mark.parametrize("model_factory", MODEL_CONSTRUCTORS)
def test_encoder_deepcopy(model_factory):
    # Not necessarily a TorchScript test, but we often want to
    # copy a module before TorchScript conversion (see e.g.
    # quantization).

    # Use a small vocab to limit memory use.
    model = model_factory(
        piece_encoder=build_wordpiece_encoder_v1(),
        with_spans=build_with_strided_spans_v1(),
        vocab_size=128,
    )
    model.initialize()
    encoder = model.get_ref("transformer").shims[0]._model
    copy.deepcopy(encoder)


@pytest.mark.slow
@pytest.mark.parametrize("model_factory", MODEL_CONSTRUCTORS)
def test_encoder_torchscript(model_factory):
    # Use a small vocab to limit memory use.
    model = model_factory(
        piece_encoder=build_wordpiece_encoder_v1(),
        with_spans=build_with_strided_spans_v1(),
        vocab_size=128,
    )
    model.initialize()
    encoder = model.get_ref("transformer").shims[0]._model
    torch.jit.script(encoder)
