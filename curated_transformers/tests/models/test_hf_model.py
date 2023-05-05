from typing import Callable
from dataclasses import dataclass
import pytest
from torch.nn import Module

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.models.curated_transformer import CuratedTransformer
from curated_transformers.models.hf_util import convert_pretrained_model_for_encoder
from curated_transformers.models.albert import AlbertEncoder
from curated_transformers.models.albert.config import AlbertConfig
from curated_transformers.models.attention import AttentionMask
from curated_transformers.models.bert import BertConfig, BertEncoder
from curated_transformers.models.roberta.config import RobertaConfig
from curated_transformers.models.roberta.encoder import RobertaEncoder

from ..conftest import TORCH_DEVICES

from ..util import torch_assertclose


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


def encoder_from_config(config: ModelConfig, hf_encoder: Module):
    encoder = config.encoder(config.config)
    transformer = CuratedTransformer(encoder)
    transformer.load_state_dict(
        convert_pretrained_model_for_encoder(transformer, hf_encoder.state_dict())
    )
    return transformer


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("model_config", TEST_MODELS)
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_model_against_hf_transformers(model_config, torch_device):
    hf_encoder = transformers.AutoModel.from_pretrained(model_config.hf_model_name).to(
        torch_device
    )
    hf_encoder.eval()

    encoder = encoder_from_config(model_config, hf_encoder)
    encoder.to(torch_device)
    encoder.eval()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config.hf_model_name
    )
    tokenization = hf_tokenizer(
        ["This is a test.", "Let's match outputs"], padding=True, return_tensors="pt"
    ).to(torch_device)
    X = tokenization["input_ids"]
    attention_mask = tokenization["attention_mask"]

    # Test with the tokenizer's attention mask
    Y_encoder = encoder(
        X, attention_mask=AttentionMask(bool_mask=attention_mask.bool())
    )
    Y_hf_encoder = hf_encoder(X, attention_mask=attention_mask)

    torch_assertclose(
        Y_encoder.last_hidden_layer_states,
        Y_hf_encoder.last_hidden_state,
    )

    # Try to infer the attention mask from padding.
    Y_encoder = encoder(X)
    torch_assertclose(
        Y_encoder.last_hidden_layer_states,
        Y_hf_encoder.last_hidden_state,
    )
