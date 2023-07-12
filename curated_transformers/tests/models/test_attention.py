import pytest
import torch

from curated_transformers.layers.attention import (
    _TORCH_SDP,
    AttentionMask,
    enable_torch_sdp,
)
from curated_transformers.models.bert.encoder import BERTEncoder
from curated_transformers.models.gpt_neox.decoder import GPTNeoXDecoder
from curated_transformers.tests.util import torch_assertclose

from ..compat import has_hf_transformers
from ..conftest import TORCH_DEVICES

VOCAB_SIZE = 1024


def test_context_manager():
    # Check default.
    assert not _TORCH_SDP.get()

    # Check context manager default.
    with enable_torch_sdp():
        assert _TORCH_SDP.get()

    # Check that the original value is restored.
    assert not _TORCH_SDP.get()

    # Check with explicit argument.
    with enable_torch_sdp(True):
        assert _TORCH_SDP.get()
    with enable_torch_sdp(False):
        assert not _TORCH_SDP.get()


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_torch_sdp(torch_device):
    model = BERTEncoder.from_hf_hub(
        name="explosion-testing/bert-test", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, VOCAB_SIZE, (2, 10), device=torch_device)
    with torch.no_grad():
        Y = model(X).last_hidden_layer_state
        with enable_torch_sdp():
            Y_sdp = model(X).last_hidden_layer_state
    torch_assertclose(Y, Y_sdp)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_torch_sdp_mask(torch_device):
    model = BERTEncoder.from_hf_hub(
        name="explosion-testing/bert-test", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, VOCAB_SIZE, (2, 10), device=torch_device)
    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = model(
            X, attention_mask=AttentionMask(mask)
        ).last_hidden_layer_state * mask.unsqueeze(-1)
        with enable_torch_sdp():
            Y_sdp = model(
                X, attention_mask=AttentionMask(mask)
            ).last_hidden_layer_state * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_sdp)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_torch_sdp_causal(torch_device):
    model = GPTNeoXDecoder.from_hf_hub(
        name="trl-internal-testing/tiny-random-GPTNeoXForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, VOCAB_SIZE, (2, 10), device=torch_device)
    with torch.no_grad():
        Y = model(X).last_hidden_layer_state
        with enable_torch_sdp():
            Y_sdp = model(X).last_hidden_layer_state
    torch_assertclose(Y, Y_sdp)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_torch_sdp_causal_with_mask(torch_device):
    model = GPTNeoXDecoder.from_hf_hub(
        name="trl-internal-testing/tiny-random-GPTNeoXForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, VOCAB_SIZE, (2, 10), device=torch_device)
    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = model(
            X, attention_mask=AttentionMask(mask)
        ).last_hidden_layer_state * mask.unsqueeze(-1)
        with enable_torch_sdp():
            Y_sdp = model(
                X, attention_mask=AttentionMask(mask)
            ).last_hidden_layer_state * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_sdp)
