import pytest
import torch

# fmt: off
from spacy_experimental.transformers.models.hf_wrapper import encoder_from_pretrained_hf_model
from spacy_experimental.transformers._compat import has_hf_transformers, transformers
from spacy_experimental.transformers.models.hf_util import SUPPORTED_HF_MODELS
# fmt: on


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_name", SUPPORTED_HF_MODELS)
def test_hf_load_roberta_weights(model_name):
    encoder = encoder_from_pretrained_hf_model(model_name)
    assert encoder


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_name", SUPPORTED_HF_MODELS)
def test_model_against_hf_transformers(model_name):
    encoder = encoder_from_pretrained_hf_model(model_name)
    encoder.eval()
    hf_encoder = transformers.AutoModel.from_pretrained(model_name)
    hf_encoder.eval()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenization = hf_tokenizer(
        ["This is a test.", "Let's match outputs"], padding=True, return_tensors="pt"
    )
    X = tokenization["input_ids"]
    attention_mask = tokenization["attention_mask"]

    # Test with the tokenizer's attention mask
    Y_encoder = encoder(X, attention_mask=attention_mask)
    Y_hf_encoder = hf_encoder(X, attention_mask=attention_mask)
    assert torch.allclose(Y_encoder.last_hidden_output, Y_hf_encoder.last_hidden_state)

    # Try to infer the attention mask from padding.
    Y_encoder = encoder(X)
    assert torch.allclose(Y_encoder.last_hidden_output, Y_hf_encoder.last_hidden_state)
