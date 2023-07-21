import pytest

from curated_transformers.models.albert import ALBERTConfig, ALBERTEncoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import assert_encoder_output_equals_hf


def test_rejects_incorrect_number_of_groups():
    config = ALBERTConfig(num_hidden_groups=5)
    with pytest.raises(ValueError, match=r"must be divisable"):
        ALBERTEncoder(config)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_encoder(torch_device):
    assert_encoder_output_equals_hf(
        ALBERTEncoder, "explosion-testing/albert-test", torch_device
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_encoder_torch_compile(torch_device):
    assert_encoder_output_equals_hf(
        ALBERTEncoder,
        "explosion-testing/albert-test",
        torch_device,
        with_torch_compile=True,
    )
