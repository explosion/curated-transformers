import pytest

from curated_transformers.generation.dolly_v2 import DollyV2Generator
from curated_transformers.generation.falcon import FalconGenerator
from curated_transformers.models import (
    AlbertEncoder,
    BertEncoder,
    CamembertEncoder,
    GPTNeoXCausalLM,
    GPTNeoXDecoder,
    RefinedWebModelCausalLM,
    RefinedWebModelDecoder,
    RobertaEncoder,
    XlmRobertaEncoder,
)
from curated_transformers.util.auto_model import (
    AutoCausalLM,
    AutoDecoder,
    AutoEncoder,
    AutoGenerator,
)


def test_auto_encoder():
    model_encoder_map = {
        "explosion-testing/bert-test": BertEncoder,
        "explosion-testing/albert-test": AlbertEncoder,
        "explosion-testing/roberta-test": RobertaEncoder,
        "explosion-testing/camembert-test": CamembertEncoder,
        "explosion-testing/xlm-roberta-test": XlmRobertaEncoder,
    }

    for name, encoder_cls in model_encoder_map.items():
        encoder = AutoEncoder.from_hf_hub(name)
        assert isinstance(encoder, encoder_cls)

    with pytest.raises(ValueError, match="Unsupported model type"):
        AutoEncoder.from_hf_hub("explosion-testing/refined-web-model-test")


def test_auto_decoder():
    model_decoder_map = {
        "explosion-testing/refined-web-model-test": RefinedWebModelDecoder,
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM": GPTNeoXDecoder,
    }

    for name, decoder_cls in model_decoder_map.items():
        decoder = AutoDecoder.from_hf_hub(name)
        assert isinstance(decoder, decoder_cls)

    with pytest.raises(ValueError, match="Unsupported model type"):
        AutoDecoder.from_hf_hub("trl-internal-testing/tiny-random-GPT2Model")


def test_auto_causal_lm():
    model_causallm_map = {
        "explosion-testing/refined-web-model-test": RefinedWebModelCausalLM,
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM": GPTNeoXCausalLM,
    }

    for name, causal_lm_cls in model_causallm_map.items():
        causal_lm = AutoCausalLM.from_hf_hub(name)
        assert isinstance(causal_lm, causal_lm_cls)

    with pytest.raises(ValueError, match="Unsupported model type"):
        AutoCausalLM.from_hf_hub("trl-internal-testing/tiny-random-GPT2Model")


@pytest.mark.slow
def test_auto_generator():
    model_causallm_map = {
        "databricks/dolly-v2-3b": DollyV2Generator,
        "tiiuae/falcon-7b": FalconGenerator,
    }

    for name, generator_cls in model_causallm_map.items():
        generator = AutoGenerator.from_hf_hub(name)
        assert isinstance(generator, generator_cls)

    with pytest.raises(ValueError, match="Unsupported generator"):
        AutoGenerator.from_hf_hub("trl-internal-testing/tiny-random-GPT2Model")
