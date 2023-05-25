import pytest
import torch

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.models.gpt_neox.causal_lm import GPTNeoXCausalLM
from curated_transformers.tests.util import torch_assertclose

from ...conftest import TORCH_DEVICES


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_causal_lm_against_hf(torch_device):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
    )
    hf_model.eval()
    hf_model.to(torch_device)

    model = GPTNeoXCausalLM.from_hf_hub(
        "trl-internal-testing/tiny-random-GPTNeoXForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X).logits
        Y_hf = hf_model(X).logits

    torch_assertclose(Y, Y_hf)
