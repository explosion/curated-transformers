import pytest
import torch

from curated_transformers.layers.attention import AttentionMask
from curated_transformers.models.llama.causal_lm import LLaMACausalLM
from curated_transformers.tests.util import torch_assertclose

from ...compat import has_hf_transformers, transformers
from ...conftest import TORCH_DEVICES


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_causal_lm_against_hf(torch_device):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        "trl-internal-testing/tiny-random-LlamaForCausalLM"
    )
    hf_model.eval()
    hf_model.to(torch_device)

    model = LLaMACausalLM.from_hf_hub(
        name="trl-internal-testing/tiny-random-LlamaForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X).logits
        Y_hf = hf_model(X).logits
    torch_assertclose(Y, Y_hf)

    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = model(X, attention_mask=AttentionMask(mask)).logits * mask.unsqueeze(-1)
        Y_hf = hf_model(X, attention_mask=mask).logits * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_hf)
