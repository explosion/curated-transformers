from typing import Callable
from dataclasses import dataclass
import pytest
import torch
from torch.nn import Module

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.models.hf_loader import build_hf_encoder_loader_v1
from curated_transformers.models.pytorch.albert import AlbertEncoder
from curated_transformers.models.pytorch.albert.config import AlbertConfig
from curated_transformers.models.pytorch.attention import AttentionMask
from curated_transformers.models.pytorch.bert import BertConfig, BertEncoder
from curated_transformers.models.pytorch.roberta.config import RobertaConfig
from curated_transformers.models.pytorch.roberta.encoder import RobertaEncoder
from curated_transformers.models.transformer_model import _pytorch_encoder


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
    model = _pytorch_encoder(encoder)
    model.init = build_hf_encoder_loader_v1(name=config.hf_model_name)
    return model


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("model_config", TEST_MODELS)
def test_hf_load_weights(model_config):
    model = encoder_from_config(model_config)
    assert model


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
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
    Y_encoder = encoder(
        X, attention_mask=AttentionMask(bool_mask=attention_mask.bool())
    )
    Y_hf_encoder = hf_encoder(X, attention_mask=attention_mask)

    assert torch.allclose(
        Y_encoder.last_hidden_layer_states, Y_hf_encoder.last_hidden_state
    )

    # Try to infer the attention mask from padding.
    Y_encoder = encoder(X)
    assert torch.allclose(
        Y_encoder.last_hidden_layer_states, Y_hf_encoder.last_hidden_state
    )
