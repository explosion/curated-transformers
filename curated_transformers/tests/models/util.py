from typing import Type

import torch

from curated_transformers.layers.attention import AttentionMask
from curated_transformers.models.hf_hub import FromHFHub

from ..compat import transformers
from ..util import torch_assertclose


def assert_encoder_output_equals_hf(
    model_class: Type[FromHFHub],
    model_name: str,
    torch_device: torch.device,
    *,
    atol=1e-5,
    rtol=1e-5
):
    model = model_class.from_hf_hub(name=model_name, device=torch_device)
    model.eval()

    for _, param in model.state_dict().items():
        assert param.device == torch_device

    hf_model = transformers.AutoModel.from_pretrained(model_name)
    hf_model.to(torch_device)
    hf_model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X).last_hidden_layer_state
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = model(
            X, attention_mask=AttentionMask(mask)
        ).last_hidden_layer_state * mask.unsqueeze(-1)
        Y_hf = hf_model(X, attention_mask=mask).last_hidden_state * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_hf)
