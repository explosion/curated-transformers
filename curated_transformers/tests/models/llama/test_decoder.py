import pytest

from curated_transformers.models.llama.decoder import LLaMADecoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import assert_decoder_output_equals_hf


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder(torch_device):
    assert_decoder_output_equals_hf(
        LLaMADecoder, "trl-internal-testing/tiny-random-LlamaForCausalLM", torch_device
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder_with_torch_compile(torch_device):
    assert_decoder_output_equals_hf(
        LLaMADecoder,
        "trl-internal-testing/tiny-random-LlamaForCausalLM",
        torch_device,
        with_torch_compile=True,
    )
