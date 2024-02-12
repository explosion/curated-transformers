import pytest

from curated_transformers.models.hf_hub.conversion import CommonHFKeys
from curated_transformers.models.mpt._hf import HFConfigKeys
from curated_transformers.models.mpt.causal_lm import MPTCausalLM

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import (
    JITMethod,
    assert_causal_lm_output_equals_hf,
    assert_model_hf_serialization_roundtrip,
)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_causal_lm(torch_device, with_torch_sdp):
    assert_causal_lm_output_equals_hf(
        MPTCausalLM,
        "explosion-testing/mpt-test",
        torch_device,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_causal_lm_with_torch_compile(torch_device, with_torch_sdp):
    assert_causal_lm_output_equals_hf(
        MPTCausalLM,
        "explosion-testing/mpt-test",
        torch_device,
        jit_method=JITMethod.TorchCompile,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_causal_lm_hf_serializtion_roundtrip(torch_device):
    assert_model_hf_serialization_roundtrip(
        MPTCausalLM,
        "explosion-testing/mpt-test",
        torch_device,
        optional_hf_config_keys={
            HFConfigKeys.LAYER_NORM_EPSILON.name,
            CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB.name,
            CommonHFKeys.HIDDEN_DROPOUT_PROB.name,
        },
    )
