import pytest

from curated_transformers.models.electra.encoder import ELECTRAEncoder

from ...compat import has_hf_transformers
from ...conftest import TORCH_DEVICES
from ..util import assert_encoder_output_equals_hf


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
@pytest.mark.parametrize(
    "model_name",
    ["explosion-testing/electra-test"],
)
def test_encoder(model_name: str, torch_device, with_torch_sdp):
    assert_encoder_output_equals_hf(
        ELECTRAEncoder,
        model_name,
        torch_device,
        with_torch_sdp=with_torch_sdp,
    )
