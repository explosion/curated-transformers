import pytest
from huggingface_hub import HfFileSystem

from curated_transformers.models import (
    ALBERTEncoder,
    BERTEncoder,
    CamemBERTEncoder,
    FalconCausalLM,
    FalconDecoder,
    GPTNeoXCausalLM,
    GPTNeoXDecoder,
    RoBERTaEncoder,
    XLMREncoder,
)
from curated_transformers.models.auto_model import (
    AutoCausalLM,
    AutoDecoder,
    AutoEncoder,
)
from curated_transformers.models.llama.causal_lm import LlamaCausalLM
from curated_transformers.models.llama.decoder import LlamaDecoder
from curated_transformers.models.mpt.causal_lm import MPTCausalLM
from curated_transformers.models.mpt.decoder import MPTDecoder
from curated_transformers.repository.fsspec import FsspecArgs


@pytest.fixture
def model_encoder_map():
    return {
        "explosion-testing/bert-test": BERTEncoder,
        "explosion-testing/albert-test": ALBERTEncoder,
        "explosion-testing/roberta-test": RoBERTaEncoder,
        "explosion-testing/camembert-test": CamemBERTEncoder,
        "explosion-testing/xlm-roberta-test": XLMREncoder,
    }


def test_auto_encoder(model_encoder_map):
    for name, encoder_cls in model_encoder_map.items():
        encoder = AutoEncoder.from_hf_hub(name=name)
        assert isinstance(encoder, encoder_cls)

    with pytest.raises(ValueError, match="Unsupported model type"):
        AutoEncoder.from_hf_hub(name="explosion-testing/falcon-test")


@pytest.mark.slow
def test_auto_encoder_fsspec(model_encoder_map):
    for name, encoder_cls in model_encoder_map.items():
        # The default revision is 'main', but we pass it anyway to test
        # that the function acceps fsspec_args.
        encoder = AutoEncoder.from_fsspec(
            fs=HfFileSystem(), model_path=name, fsspec_args=FsspecArgs(revision="main")
        )
        assert isinstance(encoder, encoder_cls)


def test_auto_decoder():
    model_decoder_map = {
        "explosion-testing/falcon-test": FalconDecoder,
        "explosion-testing/llama2-kv-sharing": LlamaDecoder,
        "explosion-testing/mpt-test": MPTDecoder,
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM": GPTNeoXDecoder,
    }

    for name, decoder_cls in model_decoder_map.items():
        decoder = AutoDecoder.from_hf_hub(name=name)
        assert isinstance(decoder, decoder_cls)

    with pytest.raises(ValueError, match="Unsupported model type"):
        AutoDecoder.from_hf_hub(name="trl-internal-testing/tiny-random-GPT2Model")


def test_auto_causal_lm():
    model_causallm_map = {
        "explosion-testing/falcon-test": FalconCausalLM,
        "explosion-testing/llama2-kv-sharing": LlamaCausalLM,
        "explosion-testing/mpt-test": MPTCausalLM,
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM": GPTNeoXCausalLM,
    }

    for name, causal_lm_cls in model_causallm_map.items():
        causal_lm = AutoCausalLM.from_hf_hub(name=name)
        assert isinstance(causal_lm, causal_lm_cls)

    with pytest.raises(ValueError, match="Unsupported model type"):
        AutoCausalLM.from_hf_hub(name="trl-internal-testing/tiny-random-GPT2Model")
