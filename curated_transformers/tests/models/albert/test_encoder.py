import pytest

from curated_transformers._compat import has_hf_transformers
from curated_transformers.models.albert import AlbertConfig, AlbertEncoder


from ...conftest import TORCH_DEVICES
from ..util import assert_encoder_output_equals_hf


def test_rejects_incorrect_number_of_groups():
    config = AlbertConfig(num_hidden_groups=5)
    with pytest.raises(ValueError, match=r"must be divisable"):
        AlbertEncoder(config)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_encoder(torch_device):
    assert_encoder_output_equals_hf(
        AlbertEncoder, "explosion-testing/albert-test", torch_device
    )
