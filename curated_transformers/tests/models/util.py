from enum import Enum
from typing import Callable, Dict, List, Tuple, Type, Union

import torch
from torch import Tensor
from torch.nn import Module

from curated_transformers.layers.attention import AttentionMask, enable_torch_sdp
from curated_transformers.layers.cache import KeyValueCache
from curated_transformers.models.hf_hub import FromHFHub
from curated_transformers.models.module import (
    CausalLMModule,
    DecoderModule,
    EncoderModule,
)
from curated_transformers.models.output import ModelOutput, ModelOutputWithCache

from ..compat import transformers
from ..utils import torch_assertclose


class DecoderWithCache(Module):
    def __init__(self, decoder: DecoderModule):
        super().__init__()
        self.inner = decoder

    def forward(self, input_ids: Tensor, cache: List[KeyValueCache]):
        return self.inner.forward(input_ids=input_ids, cache=cache, store_cache=True)


class DecoderWithPositions(Module):
    def __init__(self, decoder: DecoderModule):
        super().__init__()
        self.inner = decoder

    def forward(self, input_ids: Tensor, positions: Tensor):
        return self.inner.forward(input_ids=input_ids, positions=positions)


class JITMethod(Enum):
    Disable = 0
    TorchCompile = 1
    TorchScriptTrace = 2

    def convert(
        self, model: Module, with_torch_sdp: bool, *args
    ) -> Tuple[
        Union[Module, torch.ScriptModule],
        Callable[[Union[ModelOutput, Dict[str, torch.Tensor]]], Tensor],
    ]:
        with enable_torch_sdp(with_torch_sdp):
            if self == JITMethod.Disable:
                return model, lambda s: s
            elif self == JITMethod.TorchCompile:
                return torch.compile(model), lambda s: s
            else:
                if isinstance(model, EncoderModule):
                    cls = ModelOutput
                elif isinstance(model, DecoderModule):
                    cls = ModelOutputWithCache
                elif isinstance(model, CausalLMModule):
                    cls = ModelOutputWithCache
                return (
                    torch.jit.trace(model, tuple(args)),
                    lambda s: s,
                )


def assert_causal_lm_output_equals_hf(
    model_class: Type[FromHFHub],
    model_name: str,
    torch_device: torch.device,
    *,
    atol=1e-5,
    rtol=1e-5,
    model_revision: str = "main",
    jit_method: JITMethod = JITMethod.Disable,
    with_torch_sdp=False,
):
    orig_model = model_class.from_hf_hub(
        name=model_name, revision=model_revision, device=torch_device
    )
    orig_model.eval()

    for _, param in orig_model.state_dict().items():
        assert param.device == torch_device

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
    )
    hf_model.eval()
    hf_model.to(torch_device)

    torch.manual_seed(0)
    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)
    model, get_output = jit_method.convert(orig_model, with_torch_sdp, X_jit)

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    with torch.no_grad():
        Y = get_output(model(X))[1]
        Y_hf = hf_model(X).logits
    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    mask_jit = torch.rand_like(X_jit, dtype=torch.float32) < 0.5
    model, get_output = jit_method.convert(
        orig_model, with_torch_sdp, X_jit, AttentionMask(mask_jit)
    )

    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = get_output(model(X, attention_mask=AttentionMask(mask)))[
            1
        ] * mask.unsqueeze(-1)
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
    jit_method: JITMethod = JITMethod.Disable,
    with_torch_sdp=False,
):
    orig_model = model_class.from_hf_hub(
        name=model_name, revision=model_revision, device=torch_device
    )
    orig_model.eval()

    for _, param in orig_model.state_dict().items():
        assert param.device == torch_device

    hf_model = transformers.AutoModel.from_pretrained(
        model_name, revision=model_revision, trust_remote_code=trust_remote_code
    )
    hf_model.to(torch_device)
    hf_model.eval()

    torch.manual_seed(0)
    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)

    model, output = jit_method.convert(orig_model, with_torch_sdp, X_jit)

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    with torch.no_grad():
        mo = output(model(X))
        Y = output(model(X))[0][-1]
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    if with_cache:
        assert_decoder_with_cache_output_equals_hf(
            orig_model, hf_model, torch_device, atol, rtol, jit_method
        )

    if with_positions:
        assert_decoder_with_positions_equals_hf(
            orig_model, hf_model, torch_device, atol, rtol, jit_method
        )

    if with_mask:
        assert_with_mask_output_equals_hf(
            orig_model, hf_model, torch_device, atol, rtol, jit_method
        )


def assert_encoder_output_equals_hf(
    model_class: Type[FromHFHub],
    model_name: str,
    torch_device: torch.device,
    *,
    atol=1e-5,
    rtol=1e-5,
    jit_method: JITMethod = JITMethod.Disable,
    with_torch_sdp=False,
):
    orig_model = model_class.from_hf_hub(name=model_name, device=torch_device)
    orig_model.eval()

    for _, param in orig_model.state_dict().items():
        assert param.device == torch_device

    hf_model = transformers.AutoModel.from_pretrained(model_name)
    hf_model.to(torch_device)
    hf_model.eval()

    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)
    model, output = jit_method.convert(orig_model, with_torch_sdp, X_jit)

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)

    with torch.no_grad():
        Y = output(model(X))[0][-1]
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    assert_with_mask_output_equals_hf(
        orig_model, hf_model, torch_device, atol, rtol, jit_method
    )


def assert_decoder_with_cache_output_equals_hf(
    orig_model: DecoderModule,
    hf_model: "transformers.AutoModel",
    torch_device: torch.device,
    atol: float,
    rtol: float,
    jit_method: JITMethod,
    with_torch_sdp=False,
):
    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)
    cache_jit = orig_model(X_jit, store_cache=True).cache

    model, output = jit_method.convert(
        DecoderWithCache(orig_model),
        with_torch_sdp,
        X_jit,
        cache_jit,
    )

    _, n_heads, _, head_dim = cache_jit[0].key.shape
    empty_kv_jit = torch.zeros(
        (2, n_heads, 0, head_dim),
        dtype=cache_jit[0].key.dtype,
        device=torch_device,
    )
    empty_cache_jit = [
        KeyValueCache(empty_kv_jit, empty_kv_jit)
    ] * hf_model.config.num_hidden_layers

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    X_rest = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    with torch.no_grad():
        Y = model(X, empty_cache_jit)
        Y_hf = hf_model(X, use_cache=True)
        Y = output(model(X_rest, cache=output(Y)[1]))[0][-1]
        Y_hf = hf_model(X_rest, past_key_values=Y_hf.past_key_values).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)


def assert_with_mask_output_equals_hf(
    orig_model: Union[DecoderModule, EncoderModule],
    hf_model: "transformers.AutoModel",
    torch_device: torch.device,
    atol: float,
    rtol: float,
    jit_method: JITMethod,
    with_torch_sdp=False,
):
    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)
    mask_jit = torch.rand_like(X_jit, dtype=torch.float32) < 0.5
    model, output = jit_method.convert(
        orig_model, with_torch_sdp, X_jit, AttentionMask(mask_jit)
    )

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = output(model(X, attention_mask=AttentionMask(mask)))[0][
            -1
        ] * mask.unsqueeze(-1)
        Y_hf = hf_model(X, attention_mask=mask).last_hidden_state * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)


def assert_decoder_with_positions_equals_hf(
    orig_model: DecoderModule,
    hf_model: "transformers.AutoModel",
    torch_device: torch.device,
    atol: float,
    rtol: float,
    jit_method: JITMethod,
    with_torch_sdp=False,
):
    # NOTE: this test may break in the future because we are relying on a
    #       slicing bug, which results in more general support for positions
    #       (like our implementation):
    #
    #       https://github.com/huggingface/transformers/blob/f924df3c7e5150317ef47754a31cebf2893570ce/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L278
    #
    #       However, it is still nice to test this with arbitrary positions. If
    #       this breaks, just replace max_position_embbeddings by the sequence
    #       length.

    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)
    positions_jit = torch.randint_like(X_jit, 10)
    model, output = jit_method.convert(
        DecoderWithPositions(orig_model), with_torch_sdp, X_jit, positions_jit
    )

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    positions = torch.randint(0, 10, (2, 10), device=torch_device)
    with torch.no_grad():
        Y = output(model(X, positions=positions))[0][-1]
        Y_hf = hf_model(X, position_ids=positions).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)
