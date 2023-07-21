from typing import Type

import torch

from curated_transformers.layers.attention import AttentionMask
from curated_transformers.models.hf_hub import FromHFHub

from ..compat import transformers
from ..util import torch_assertclose


def assert_causal_lm_output_equals_hf(
    model_class: Type[FromHFHub],
    model_name: str,
    torch_device: torch.device,
    *,
    atol=1e-5,
    rtol=1e-5,
    model_revision: str = "main",
    with_torch_compile: bool = False,
):
    model = model_class.from_hf_hub(
        name=model_name, revision=model_revision, device=torch_device
    )
    model.eval()

    for _, param in model.state_dict().items():
        assert param.device == torch_device

    if with_torch_compile:
        model = torch.compile(model)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
    )
    hf_model.eval()
    hf_model.to(torch_device)

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X).logits
        Y_hf = hf_model(X).logits
    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = model(X, attention_mask=AttentionMask(mask)).logits * mask.unsqueeze(-1)
        Y_hf = hf_model(X, attention_mask=mask).logits * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)


def assert_decoder_output_equals_hf(
    model_class: Type[FromHFHub],
    model_name: str,
    torch_device: torch.device,
    *,
    atol=1e-5,
    rtol=1e-5,
    model_revision: str = "main",
    trust_remote_code: bool = False,
    with_cache: bool = True,
    with_positions: bool = True,
    with_mask: bool = True,
    with_torch_compile: bool = False,
):
    model = model_class.from_hf_hub(
        name=model_name, revision=model_revision, device=torch_device
    )
    model.eval()

    for _, param in model.state_dict().items():
        assert param.device == torch_device

    if with_torch_compile:
        model = torch.compile(model)

    hf_model = transformers.AutoModel.from_pretrained(
        model_name, revision=model_revision, trust_remote_code=trust_remote_code
    )
    hf_model.to(torch_device)
    hf_model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = model(X).last_hidden_layer_state
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    if with_cache:
        X_rest = torch.randint(
            0, hf_model.config.vocab_size, (2, 10), device=torch_device
        )

        with torch.no_grad():
            Y = model(X, store_cache=True)
            Y_hf = hf_model(X, use_cache=True)
            Y = model(X_rest, cache=Y.cache).last_hidden_layer_state
            Y_hf = hf_model(
                X_rest, past_key_values=Y_hf.past_key_values
            ).last_hidden_state

        torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    if with_positions:
        # NOTE: this test may break in the future because we are relying on a
        #       slicing bug, which results in more general support for positions
        #       (like our implementation):
        #
        #       https://github.com/huggingface/transformers/blob/f924df3c7e5150317ef47754a31cebf2893570ce/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L278
        #
        #       However, it is still nice to test this with arbitrary positions. If
        #       this breaks, just replace max_position_embbeddings by the sequence
        #       length.
        positions = torch.randint(0, 10, (2, 10), device=torch_device)

        with torch.no_grad():
            Y = model(X, positions=positions).last_hidden_layer_state
            Y_hf = hf_model(X, position_ids=positions).last_hidden_state

        torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    if with_mask:
        mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
        with torch.no_grad():
            Y = model(
                X, attention_mask=AttentionMask(mask)
            ).last_hidden_layer_state * mask.unsqueeze(-1)
            Y_hf = hf_model(X, attention_mask=mask).last_hidden_state * mask.unsqueeze(
                -1
            )
        torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)


def assert_encoder_output_equals_hf(
    model_class: Type[FromHFHub],
    model_name: str,
    torch_device: torch.device,
    *,
    atol=1e-5,
    rtol=1e-5,
    with_torch_compile: bool = False,
):
    model = model_class.from_hf_hub(name=model_name, device=torch_device)
    model.eval()

    for _, param in model.state_dict().items():
        assert param.device == torch_device

    if with_torch_compile:
        model = torch.compile(model)

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
