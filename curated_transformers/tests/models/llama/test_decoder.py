import pytest

from curated_transformers.models.llama.decoder import LLaMADecoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import assert_decoder_output_equals_hf

LLAMA_TEST_MODELS = [
    "trl-internal-testing/tiny-random-LlamaForCausalLM",
    "explosion-testing/llama2-fewer-kv-heads",
    "explosion-testing/llama2-kv-sharing",
]


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model", LLAMA_TEST_MODELS)
def test_decoder(torch_device, model):
    assert_decoder_output_equals_hf(LLaMADecoder, model, torch_device)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model", LLAMA_TEST_MODELS)
def test_decoder_with_torch_compile(torch_device, model):
    assert_decoder_output_equals_hf(
        LLaMADecoder,
        model,
        torch_device,
        with_torch_compile=True,
    )
