from typing import Callable
from dataclasses import dataclass
from functools import partial
from curated_transformers.models.albert import AlbertEncoder
from curated_transformers.models.albert.config import AlbertConfig
from curated_transformers.models.bert import BertConfig, BertEncoder
from curated_transformers.models.roberta.config import RobertaConfig
from curated_transformers.models.roberta.encoder import RobertaEncoder
import pytest
import torch
from torch.nn import Module

# fmt: off
from curated_transformers.models.hf_wrapper import build_hf_transformer_encoder_v1, build_hf_encoder_loader
from curated_transformers._compat import has_hf_transformers, transformers
# fmt: on


@dataclass
class ModelConfig:
    config: BertConfig
    encoder: Callable[[BertConfig], Module]
    hf_model_name: str


TEST_MODELS = [
    ModelConfig(AlbertConfig(vocab_size=30000), AlbertEncoder, "albert-base-v2"),
    ModelConfig(BertConfig(vocab_size=28996), BertEncoder, "bert-base-cased"),
    ModelConfig(RobertaConfig(), RobertaEncoder, "roberta-base"),
    ModelConfig(RobertaConfig(vocab_size=250002), RobertaEncoder, "xlm-roberta-base"),
]


def encoder_from_config(config: ModelConfig):
    encoder = config.encoder(config.config)
    return build_hf_transformer_encoder_v1(
        encoder, init=build_hf_encoder_loader(name=config.hf_model_name)
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_config", TEST_MODELS)
def test_hf_load_weights(model_config):
    model = encoder_from_config(model_config)
    assert model


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_config", TEST_MODELS)
def test_model_against_hf_transformers(model_config):
    model = encoder_from_config(model_config)
    model.initialize()
    encoder = model.shims[0]._model
    encoder.eval()
    hf_encoder = transformers.AutoModel.from_pretrained(model_config.hf_model_name)
    hf_encoder.eval()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config.hf_model_name
    )
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
