import pytest

# fmt: off
from spacy_experimental.transformers.models.hf_wrapper import encoder_from_pretrained_hf_model
from spacy_experimental.transformers.models.hf_util import has_hf_transformers, SUPPORTED_HF_MODELS
# fmt: on


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_name", SUPPORTED_HF_MODELS)
def test_hf_load_roberta_weights(model_name):
    encoder = encoder_from_pretrained_hf_model(model_name)
    assert encoder
