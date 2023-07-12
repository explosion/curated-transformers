import pytest

from curated_transformers.models.bert.encoder import BERTEncoder

from ..compat import has_hf_transformers
from ..conftest import TORCH_DEVICES
from .util import assert_encoder_output_equals_hf


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_sharded_model_checkpoints(torch_device):
    assert_encoder_output_equals_hf(
        BERTEncoder, "explosion-testing/bert-test-sharded", torch_device
    )
