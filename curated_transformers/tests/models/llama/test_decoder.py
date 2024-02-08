import pytest

from curated_transformers.models.hf_hub.conversion import CommonHFKeys
from curated_transformers.models.llama._hf import HFConfigKeys
from curated_transformers.models.llama.decoder import LlamaDecoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import (
    JITMethod,
    assert_decoder_output_equals_hf,
    assert_model_hf_serialization_roundtrip,
)

LLAMA_TEST_MODELS = [
    "trl-internal-testing/tiny-random-LlamaForCausalLM",
    "explosion-testing/llama2-fewer-kv-heads",
    "explosion-testing/llama2-kv-sharing",
]


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model", LLAMA_TEST_MODELS)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_decoder(torch_device, model, with_torch_sdp):
    assert_decoder_output_equals_hf(
        LlamaDecoder, model, torch_device, with_torch_sdp=with_torch_sdp
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model", LLAMA_TEST_MODELS)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_decoder_with_torch_compile(torch_device, model, with_torch_sdp):
    assert_decoder_output_equals_hf(
        LlamaDecoder,
        model,
        torch_device,
        jit_method=JITMethod.TorchCompile,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("model", LLAMA_TEST_MODELS)
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder_hf_serializtion_roundtrip(model, torch_device):
    assert_model_hf_serialization_roundtrip(
        LlamaDecoder,
        model,
        torch_device,
        optional_hf_config_keys={
            HFConfigKeys.NUM_KEY_VALUE_HEADS.name,
            CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB.name,
            CommonHFKeys.HIDDEN_DROPOUT_PROB.name,
        },
    )
