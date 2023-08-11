import pytest
from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

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


def test_download_to_cache():
    BERTEncoder.download_to_cache(
        name="explosion-testing/bert-test-sharded",
        revision="9aa9cc67d787328ced8c8f81b6b52cc3bd036e8b",
    )

    expected_checkpoints = [
        "pytorch_model-00001-of-00007.bin",
        "pytorch_model-00002-of-00007.bin",
        "pytorch_model-00003-of-00007.bin",
        "pytorch_model-00004-of-00007.bin",
        "pytorch_model-00005-of-00007.bin",
        "pytorch_model-00006-of-00007.bin",
        "pytorch_model-00007-of-00007.bin",
    ]
    for name in expected_checkpoints:
        assert (
            try_to_load_from_cache(
                repo_id="explosion-testing/bert-test-sharded",
                filename=name,
                revision="9aa9cc67d787328ced8c8f81b6b52cc3bd036e8b",
            )
            != _CACHED_NO_EXIST
        )
