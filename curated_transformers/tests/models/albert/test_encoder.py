import pytest

from curated_transformers.models.albert import ALBERTConfig, ALBERTEncoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import (
    JITMethod,
    assert_encoder_output_equals_hf,
    assert_model_hf_serialization_roundtrip,
)


def test_rejects_incorrect_number_of_groups():
    config = ALBERTConfig(n_hidden_groups=5)
    with pytest.raises(ValueError, match=r"must be divisable"):
        ALBERTEncoder(config)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_encoder(torch_device, with_torch_sdp):
    assert_encoder_output_equals_hf(
        ALBERTEncoder,
        "explosion-testing/albert-test",
        torch_device,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_encoder_with_torch_compile(torch_device, with_torch_sdp):
    assert_encoder_output_equals_hf(
        ALBERTEncoder,
        "explosion-testing/albert-test",
        torch_device,
        jit_method=JITMethod.TorchCompile,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_encoder_hf_serializtion_roundtrip(torch_device):
    assert_model_hf_serialization_roundtrip(
        ALBERTEncoder, "explosion-testing/albert-test", torch_device
    )
