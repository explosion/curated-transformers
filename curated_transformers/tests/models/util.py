from typing import Type
import torch

from curated_transformers._compat import transformers
from curated_transformers.models.hf_hub import FromPretrainedHFModel


from ..util import torch_assertclose


def assert_encoder_output_equals_hf(
    model_class: Type[FromPretrainedHFModel],
    model_name: str,
    torch_device: torch.device,
    *,
    atol=1e-5,
    rtol=1e-5
):
    hf_model = transformers.AutoModel.from_pretrained(model_name)
    hf_model.to(torch_device)
    hf_model.eval()

    model = model_class.from_hf_hub(model_name)
    model.to(torch_device)
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X).last_hidden_layer_states
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)
