import pytest
import torch

# fmt: off
from curated_transformers.models.hf_wrapper import roberta_encoder_from_pretrained_hf_model
from curated_transformers.models.hf_wrapper import bert_encoder_from_pretrained_hf_model
from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.models.hf_util import SUPPORTED_BERT_MODELS, SUPPORTED_ROBERTA_MODELS
# fmt: on


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_name", SUPPORTED_ROBERTA_MODELS)
def test_hf_load_roberta_weights(model_name):
    encoder = roberta_encoder_from_pretrained_hf_model(model_name)
    assert encoder


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_name", SUPPORTED_BERT_MODELS)
def test_hf_load_bert_weights(model_name):
    encoder = bert_encoder_from_pretrained_hf_model(model_name)
    assert encoder


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_name", SUPPORTED_BERT_MODELS)
def test_bert_model_against_hf_transformers(model_name):
    encoder = bert_encoder_from_pretrained_hf_model(model_name)
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

    assert torch.allclose(Y_encoder.last_hidden_state, Y_hf_encoder.last_hidden_state)

    # Try to infer the attention mask from padding.
    Y_encoder = encoder(X)
    assert torch.allclose(Y_encoder.last_hidden_state, Y_hf_encoder.last_hidden_state)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_name", SUPPORTED_ROBERTA_MODELS)
def test_roberta_model_against_hf_transformers(model_name):
    encoder = roberta_encoder_from_pretrained_hf_model(model_name)
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
    assert torch.allclose(Y_encoder.last_hidden_state, Y_hf_encoder.last_hidden_state)

    # Try to infer the attention mask from padding.
    Y_encoder = encoder(X)
    assert torch.allclose(Y_encoder.last_hidden_state, Y_hf_encoder.last_hidden_state)
