from enum import Enum
from typing import Callable, Dict, List, Set, Tuple, Type, Union

import torch
from huggingface_hub import HfFileSystem
from torch import Tensor
from torch.nn import Module

from curated_transformers.layers.attention import AttentionMask, enable_torch_sdp
from curated_transformers.layers.cache import KeyValueCache
from curated_transformers.models.hf_hub import FromHF
from curated_transformers.models.module import (
    CausalLMModule,
    DecoderModule,
    EncoderModule,
    TransformerModule,
)
from curated_transformers.models.output import ModelOutput, ModelOutputWithCache
from curated_transformers.repository.hf_hub import HfHubRepository
from curated_transformers.repository.repository import ModelRepository

from ..compat import transformers
from ..utils import torch_assertclose


class DecoderWithCache(Module):
    def __init__(self, decoder: DecoderModule):
        super().__init__()
        self.inner = decoder

    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        cache: List[KeyValueCache],
    ):
        return self.inner.forward(
            piece_ids=piece_ids,
            attention_mask=attention_mask,
            cache=cache,
            store_cache=True,
        )


class DecoderWithPositions(Module):
    def __init__(self, decoder: DecoderModule):
        super().__init__()
        self.inner = decoder

    def forward(
        self, piece_ids: Tensor, attention_mask: AttentionMask, positions: Tensor
    ):
        return self.inner.forward(
            piece_ids=piece_ids, attention_mask=attention_mask, positions=positions
        )


class JITMethod(Enum):
    Disable = 0
    TorchCompile = 1

    def convert(self, model: Module, with_torch_sdp: bool, *args) -> Tuple[
        Module,
        Callable[[Union[ModelOutput, Dict[str, torch.Tensor]]], Tensor],
    ]:
        with enable_torch_sdp(with_torch_sdp):
            if self == JITMethod.TorchCompile:
                return torch.compile(model), lambda s: s
            else:  # JITMethod.Disable
                return model, lambda s: s


def assert_causal_lm_output_equals_hf(
    model_class: Type[FromHF],
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

    check_params_buffers(orig_model, torch_device)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
    )
    hf_model.eval()
    hf_model.to(torch_device)

    torch.manual_seed(0)
    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)
    mask_jit = torch.ones_like(X_jit, dtype=torch.bool)
    model, get_output = jit_method.convert(
        orig_model, with_torch_sdp, X_jit, AttentionMask(mask_jit)
    )

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    mask = torch.ones_like(X, dtype=torch.bool)
    with torch.no_grad():
        Y = get_output(model(X, AttentionMask(mask))).logits
        Y_hf = hf_model(X).logits
    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    mask_jit = torch.rand_like(X_jit, dtype=torch.float32) < 0.5
    model, get_output = jit_method.convert(
        orig_model, with_torch_sdp, X_jit, AttentionMask(mask_jit)
    )

    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = get_output(model(X, AttentionMask(mask))).logits * mask.unsqueeze(-1)
        Y_hf = hf_model(X, attention_mask=mask).logits * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)


def assert_decoder_output_equals_hf(
    model_class: Type[FromHF],
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
    check_config=True,
):
    orig_model = model_class.from_hf_hub(
        name=model_name, revision=model_revision, device=torch_device
    )
    orig_model.eval()

    check_params_buffers(orig_model, torch_device)

    hf_model = transformers.AutoModel.from_pretrained(
        model_name, revision=model_revision, trust_remote_code=trust_remote_code
    )
    hf_model.to(torch_device)
    hf_model.eval()

    torch.manual_seed(0)
    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)
    mask_jit = torch.ones_like(X_jit, dtype=torch.bool)

    model, output = jit_method.convert(
        orig_model, with_torch_sdp, X_jit, AttentionMask(mask_jit)
    )

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    mask = torch.ones_like(X, dtype=torch.bool)
    with torch.no_grad():
        Y = output(model(X, AttentionMask(mask))).last_hidden_layer_state
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

    if check_config and jit_method == JITMethod.Disable:
        assert_model_config(model, Y)


def assert_encoder_output_equals_hf(
    model_class: Type[FromHF],
    model_name: str,
    torch_device: torch.device,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    jit_method: JITMethod = JITMethod.Disable,
    with_fsspec: bool = False,
    with_torch_sdp: bool = False,
    check_config=True,
):
    if with_fsspec:
        orig_model = model_class.from_fsspec(
            fs=HfFileSystem(), model_path=model_name, device=torch_device
        )
    else:
        orig_model = model_class.from_hf_hub(name=model_name, device=torch_device)
    orig_model.eval()

    check_params_buffers(orig_model, torch_device)

    hf_model = transformers.AutoModel.from_pretrained(model_name)
    hf_model.to(torch_device)
    hf_model.eval()

    X_jit = torch.randint(0, hf_model.config.vocab_size, (3, 5), device=torch_device)
    mask_jit = torch.ones_like(X_jit, dtype=torch.bool)
    model, output = jit_method.convert(
        orig_model, with_torch_sdp, X_jit, AttentionMask(mask_jit)
    )

    torch.manual_seed(0)
    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    mask = torch.ones_like(X, dtype=torch.bool)

    with torch.no_grad():
        Y = output(model(X, AttentionMask(mask))).last_hidden_layer_state
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)

    assert_with_mask_output_equals_hf(
        orig_model, hf_model, torch_device, atol, rtol, jit_method
    )

    if check_config and jit_method == JITMethod.Disable:
        assert_model_config(model, Y)


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
    mask_jit = torch.ones_like(X_jit, dtype=torch.bool)
    with torch.no_grad():
        cache_jit = orig_model(X_jit, AttentionMask(mask_jit), store_cache=True).cache

    model, output = jit_method.convert(
        DecoderWithCache(orig_model),
        with_torch_sdp,
        X_jit,
        AttentionMask(torch.concat([mask_jit, mask_jit], dim=1)),
        cache_jit,
    )

    _, n_heads, _, head_width = cache_jit[0].key.shape
    empty_kv_jit = torch.zeros(
        (2, n_heads, 0, head_width),
        dtype=cache_jit[0].key.dtype,
        device=torch_device,
    )
    empty_cache_jit = [
        KeyValueCache(key=empty_kv_jit, value=empty_kv_jit)
    ] * hf_model.config.num_hidden_layers

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    mask = torch.ones_like(X, dtype=torch.bool)
    X_rest = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    mask_rest = torch.cat([mask, torch.ones_like(X_rest, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        Y = model(X, AttentionMask(mask), empty_cache_jit)
        Y_hf = hf_model(X, use_cache=True)
        Y = output(
            model(X_rest, AttentionMask(mask_rest), cache=output(Y).cache)
        ).last_hidden_layer_state
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
        Y = output(
            model(X, AttentionMask(mask))
        ).last_hidden_layer_state * mask.unsqueeze(-1)
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
    mask_jit = torch.ones_like(X_jit, dtype=torch.bool)
    positions_jit = torch.randint_like(X_jit, 10)
    model, output = jit_method.convert(
        DecoderWithPositions(orig_model),
        with_torch_sdp,
        X_jit,
        AttentionMask(mask_jit),
        positions_jit,
    )

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10), device=torch_device)
    mask = torch.ones_like(X, dtype=torch.bool)
    positions = torch.randint(0, 10, (2, 10), device=torch_device)
    with torch.no_grad():
        Y = output(
            model(X, AttentionMask(mask), positions=positions)
        ).last_hidden_layer_state
        Y_hf = hf_model(X, position_ids=positions).last_hidden_state

    torch_assertclose(Y, Y_hf, atol=atol, rtol=rtol)


def assert_model_config(model: TransformerModule, model_output: Tensor):
    assert isinstance(model, TransformerModule)
    config = model.config

    hidden_width = model_output.size(-1)
    assert config.layer.feedforward.hidden_width == hidden_width


def assert_model_hf_serialization_roundtrip(
    model_class: Type[FromHF],
    model_name: str,
    torch_device: torch.device,
    *,
    model_revision: str = "main",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    trust_remote_code: bool = False,
    optional_hf_config_keys: Set[str] = set(),
):
    orig_model = model_class.from_hf_hub(
        name=model_name,
        revision=model_revision,
        device=torch_device,
    )
    orig_model.eval()

    check_params_buffers(orig_model, torch_device)

    auto_cls = (
        transformers.AutoModelForCausalLM
        if isinstance(orig_model, CausalLMModule)
        else transformers.AutoModel
    )

    hf_model = auto_cls.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    hf_model.to(torch_device)
    hf_model.eval()

    hf_model_statedict = hf_model.state_dict()
    orig_model_hf_statedict = orig_model.state_dict_to_hf(orig_model.state_dict())
    for name in orig_model_hf_statedict.keys():
        assert name in hf_model_statedict.keys(), f"{name} not found in HF state dict"
        torch_assertclose(orig_model_hf_statedict[name], hf_model_statedict[name])

    hf_config = ModelRepository(
        HfHubRepository(name=model_name, revision=model_revision)
    ).model_config()
    orig_model_hf_config = orig_model.config_to_hf(orig_model.config)

    # These won't be the same anyway, so we can skip them.
    ignored_keys = ("architectures", "transformers_version")

    for k, v in orig_model_hf_config.items():
        if k in ignored_keys:
            continue
        elif k in optional_hf_config_keys and k not in hf_config:
            continue
        assert k in hf_config, f"Key '{k}' is missing in the Hugging Face model config"
        assert (
            hf_config[k] == v
        ), f"Key '{k}' value '{v}' is different in the Hugging Face model config ('{hf_config[k]}')"


def check_params_buffers(model: Module, device: torch.device):
    """
    Check that parameters/buffers are placed on the correct device and that
    parameters are leaf nodes.
    """
    for buffer in model.buffers():
        assert buffer.device == device

    for param in model.parameters():
        assert param.device == device
        assert param.is_leaf
