import pytest

from curated_transformers.models.camembert.encoder import CamemBERTEncoder

from ...compat import has_hf_transformers
from ...conftest import TORCH_DEVICES
from ..util import assert_encoder_output_equals_hf


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_encoder(torch_device):
    assert_encoder_output_equals_hf(
        CamemBERTEncoder, "explosion-testing/camembert-test", torch_device
    )
