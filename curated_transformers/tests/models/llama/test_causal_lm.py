import pytest

from curated_transformers.models.llama.causal_lm import LLaMACausalLM

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import JITMethod, assert_causal_lm_output_equals_hf

LLAMA_TEST_MODELS = [
    "trl-internal-testing/tiny-random-LlamaForCausalLM",
    "explosion-testing/llama2-fewer-kv-heads",
    "explosion-testing/llama2-kv-sharing",
]


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
@pytest.mark.parametrize("model", LLAMA_TEST_MODELS)
def test_causal_lm(torch_device, model, with_torch_sdp):
    assert_causal_lm_output_equals_hf(
        LLaMACausalLM,
        model,
        torch_device,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model", LLAMA_TEST_MODELS)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_causal_lm_torch_compile(torch_device, model, with_torch_sdp):
    assert_causal_lm_output_equals_hf(
        LLaMACausalLM,
        model,
        torch_device,
        jit_method=JITMethod.TorchCompile,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model", LLAMA_TEST_MODELS)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_causal_lm_with_torchscript_trace(torch_device, model, with_torch_sdp):
    assert_causal_lm_output_equals_hf(
        LLaMACausalLM,
        model,
        torch_device,
        jit_method=JITMethod.TorchScriptTrace,
        with_torch_sdp=with_torch_sdp,
    )
