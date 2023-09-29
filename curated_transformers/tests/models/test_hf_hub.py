from pathlib import Path

import pytest
from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

from curated_transformers.models.bert.encoder import BERTEncoder
from curated_transformers.repository.hf_hub import HfHubRepository
from curated_transformers.repository.repository import ModelRepository
from curated_transformers.util.serde.checkpoint import ModelCheckpointType

from ..compat import has_hf_transformers, has_safetensors
from ..conftest import TORCH_DEVICES
from .util import assert_encoder_output_equals_hf


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_sharded_model_checkpoints(torch_device):
    assert_encoder_output_equals_hf(
        BERTEncoder, "explosion-testing/bert-test-sharded", torch_device
    )


def test_from_hf_hub_to_cache():
    BERTEncoder.from_hf_hub_to_cache(
        name="explosion-testing/bert-test-caching",
        revision="96a29a07d0fa4c24fd2675521add643e3c2581fc",
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
                repo_id="explosion-testing/bert-test-caching",
                filename=name,
                revision="96a29a07d0fa4c24fd2675521add643e3c2581fc",
            )
            != _CACHED_NO_EXIST
        )


@pytest.mark.skipif(has_safetensors, reason="cannot run with huggingface safetensors")
def test_checkpoint_type_without_safetensors():
    # By default, we expect the torch checkpoint to be loaded
    # even if the safetensor checkpoints are present
    # (as long as the library is not installed).
    ckp_paths, ckp_type = ModelRepository(
        HfHubRepository("explosion-testing/safetensors-test", revision="main")
    ).model_checkpoints()
    assert len(ckp_paths) == 1
    assert ckp_paths[0].path is not None
    assert Path(ckp_paths[0].path).suffix == ".bin"
    assert ckp_type == ModelCheckpointType.PYTORCH_STATE_DICT


@pytest.mark.skipif(not has_safetensors, reason="requires huggingface safetensors")
def test_checkpoint_type_with_safetensors():
    # Since the safetensors library is installed, we should be
    # loading from those checkpoints.
    repo = ModelRepository(
        HfHubRepository("explosion-testing/safetensors-test", revision="main")
    )
    ckp_paths, ckp_type = repo.model_checkpoints()
    assert len(ckp_paths) == 1
    assert Path(ckp_paths[0].path).suffix == ".safetensors"
    assert ckp_type == ModelCheckpointType.SAFE_TENSORS

    encoder = BERTEncoder.from_hf_hub(name="explosion-testing/safetensors-test")


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_fsspec(torch_device):
    assert_encoder_output_equals_hf(
        BERTEncoder,
        "explosion-testing/bert-test",
        torch_device,
        with_fsspec=True,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_fsspec_sharded(torch_device):
    assert_encoder_output_equals_hf(
        BERTEncoder,
        "explosion-testing/bert-test-sharded",
        torch_device,
        with_fsspec=True,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_fsspec_safetensors(torch_device):
    assert_encoder_output_equals_hf(
        BERTEncoder,
        "explosion-testing/safetensors-test",
        torch_device,
        with_fsspec=True,
    )
