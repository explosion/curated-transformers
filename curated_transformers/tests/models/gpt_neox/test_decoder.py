import pytest

from curated_transformers.models.gpt_neox.decoder import GPTNeoXDecoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import assert_decoder_output_equals_hf


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder(torch_device):
    assert_decoder_output_equals_hf(
        GPTNeoXDecoder,
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM",
        torch_device,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder_with_torch_compile(torch_device):
    # NOTE: testing with cache/positions fails. All outputs seems ok until
    # the concatenation of the vectors with/without rotary embeddings:
    #
    # https://github.com/explosion/curated-transformers/blob/6de9b9828b1d6f8f52dba8b87b4e00a9732a2d5f/curated_transformers/layers/embeddings.py#L308
    assert_decoder_output_equals_hf(
        GPTNeoXDecoder,
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM",
        torch_device,
        with_cache=False,
        with_positions=False,
        with_torch_compile=True,
    )
