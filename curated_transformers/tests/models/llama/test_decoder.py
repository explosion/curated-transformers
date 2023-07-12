import pytest
import torch

from curated_transformers.layers.attention import AttentionMask
from curated_transformers.models.llama.decoder import LLaMADecoder
from curated_transformers.tests.util import torch_assertclose

from ...compat import has_hf_transformers, transformers
from ...conftest import TORCH_DEVICES


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder(torch_device):
    hf_model = transformers.AutoModel.from_pretrained(
        "trl-internal-testing/tiny-random-LlamaForCausalLM"
    )
    hf_model.to(torch_device)
    hf_model.eval()

    model = LLaMADecoder.from_hf_hub(
        name="trl-internal-testing/tiny-random-LlamaForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X).last_hidden_layer_state
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder_with_cache(torch_device):
    hf_model = transformers.AutoModel.from_pretrained(
        "trl-internal-testing/tiny-random-LlamaForCausalLM"
    )
    hf_model.to(torch_device)
    hf_model.eval()

    model = LLaMADecoder.from_hf_hub(
        name="trl-internal-testing/tiny-random-LlamaForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    X_rest = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X, store_cache=True)
        Y_hf = hf_model(X, use_cache=True)
        Y = model(X_rest, cache=Y.cache).last_hidden_layer_state
        Y_hf = hf_model(X_rest, past_key_values=Y_hf.past_key_values).last_hidden_state

    torch_assertclose(Y, Y_hf)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder_with_positions(torch_device):
    hf_model = transformers.AutoModel.from_pretrained(
        "trl-internal-testing/tiny-random-LlamaForCausalLM"
    )
    hf_model.to(torch_device)
    hf_model.eval()

    model = LLaMADecoder.from_hf_hub(
        name="trl-internal-testing/tiny-random-LlamaForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    positions = torch.randint(0, 10, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X, positions=positions).last_hidden_layer_state
        Y_hf = hf_model(X, position_ids=positions).last_hidden_state
    torch_assertclose(Y, Y_hf)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_decoder_with_mask(torch_device):
    hf_model = transformers.AutoModel.from_pretrained(
        "trl-internal-testing/tiny-random-LlamaForCausalLM"
    )
    hf_model.to(torch_device)
    hf_model.eval()

    model = LLaMADecoder.from_hf_hub(
        name="trl-internal-testing/tiny-random-LlamaForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = model(
            X, attention_mask=AttentionMask(mask)
        ).last_hidden_layer_state * mask.unsqueeze(-1)
        Y_hf = hf_model(X, attention_mask=mask).last_hidden_state * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_hf)
