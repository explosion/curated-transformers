import pytest

from curated_transformers.models.hf_hub.conversion import CommonHFKeys
from curated_transformers.models.mpt._hf import HFConfigKeys
from curated_transformers.models.mpt.decoder import MPTDecoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import (
    JITMethod,
    assert_decoder_output_equals_hf,
    assert_model_hf_serialization_roundtrip,
)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_decoder(torch_device, with_torch_sdp):
    assert_decoder_output_equals_hf(
        MPTDecoder,
        "explosion-testing/mpt-test",
        torch_device,
        # HF model does not support position IDs.
        with_positions=False,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_decoder_with_torch_compile(torch_device, with_torch_sdp):
    # NOTE: testing with cache/positions fails. All outputs seems ok until
    # the concatenation of the vectors with/without rotary embeddings:
    #
    # https://github.com/explosion/curated-transformers/blob/6de9b9828b1d6f8f52dba8b87b4e00a9732a2d5f/curated_transformers/layers/embeddings.py#L308
    assert_decoder_output_equals_hf(
        MPTDecoder,
        "explosion-testing/mpt-test",
        torch_device,
        with_cache=False,
        with_positions=False,
        jit_method=JITMethod.TorchCompile,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder_hf_serializtion_roundtrip(torch_device):
    assert_model_hf_serialization_roundtrip(
        MPTDecoder,
        "explosion-testing/mpt-test",
        torch_device,
        optional_hf_config_keys={
            HFConfigKeys.LAYER_NORM_EPSILON.name,
            CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB.name,
            CommonHFKeys.HIDDEN_DROPOUT_PROB.name,
        },
    )
