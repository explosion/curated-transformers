import pytest

from curated_transformers.models.llama.decoder import LLaMADecoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import JITMethod, assert_decoder_output_equals_hf


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_decoder(torch_device, with_torch_sdp):
    assert_decoder_output_equals_hf(
        LLaMADecoder,
        "trl-internal-testing/tiny-random-LlamaForCausalLM",
        torch_device,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_decoder_with_torch_compile(torch_device, with_torch_sdp):
    assert_decoder_output_equals_hf(
        LLaMADecoder,
        "trl-internal-testing/tiny-random-LlamaForCausalLM",
        torch_device,
        jit_method=JITMethod.TorchCompile,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_decoder_with_torchscript_trace(torch_device, with_torch_sdp):
    assert_decoder_output_equals_hf(
        LLaMADecoder,
        "trl-internal-testing/tiny-random-LlamaForCausalLM",
        torch_device,
        jit_method=JITMethod.TorchScriptTrace,
        with_torch_sdp=with_torch_sdp,
    )
