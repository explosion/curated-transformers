import pytest

from curated_transformers.models.gpt_neox.causal_lm import GPTNeoXCausalLM

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import assert_causal_lm_output_equals_hf


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_causal_lm(torch_device):
    assert_causal_lm_output_equals_hf(
        GPTNeoXCausalLM,
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM",
        torch_device,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_causal_lm_torch_compile(torch_device):
    assert_causal_lm_output_equals_hf(
        GPTNeoXCausalLM,
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM",
        torch_device,
        with_torch_compile=True,
    )
