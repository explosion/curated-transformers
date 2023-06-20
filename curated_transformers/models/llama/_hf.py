import re
from typing import Any, Mapping

from torch import Tensor

from ..module import DecoderModule
from .config import LLaMAConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]


def convert_hf_config(hf_config: Any) -> LLaMAConfig:
    # Handle config options that are not set in all models.
    extra_kwargs = {k: hf_config[k] for k in EXTRA_KWARG_KEYS if k in hf_config}

    return LLaMAConfig(
        hidden_act=hf_config["hidden_act"],
        hidden_width=hf_config["hidden_size"],
        intermediate_width=hf_config["intermediate_size"],
        rms_norm_eps=hf_config["rms_norm_eps"],
        num_attention_heads=hf_config["num_attention_heads"],
        num_hidden_layers=hf_config["num_hidden_layers"],
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        vocab_size=hf_config["vocab_size"],
        **extra_kwargs
    )


def convert_hf_state_dict(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Convert state dict from HF paramater naming to ours.
    The function is insensitive to prefixes, to allow loading
    both the decoder and the full LM."""
    if issubclass(cls, DecoderModule):
        stripped_params = {re.sub(r"^model\.", "", k): v for k, v in params.items()}
    else:
        stripped_params = {
            re.sub(r"^model\.", "decoder.", k): v for k, v in params.items()
        }

    out = {}
    for name, parameter in stripped_params.items():
        # Attention
        name = re.sub(r"\.self_attn", r".mha", name)
        name = re.sub(r"\.q_proj", r".query", name)
        name = re.sub(r"\.k_proj", r".key", name)
        name = re.sub(r"\.v_proj", r".value", name)
        name = re.sub(r"\.o_proj", r".output", name)

        # Pointwise feedforward
        name = re.sub(r"\.mlp", r".ffn", name)
        name = re.sub(r"\.up_proj", r".intermediate", name)
        name = re.sub(r"\.down_proj", r".output", name)
        name = re.sub(r"\.gate_proj", r".gate", name)

        # RMS norms
        name = re.sub(r"\.input_layernorm", r".attn_rms_norm", name)
        name = re.sub(r"\.post_attention_layernorm", r".ffn_rms_norm", name)
        name = re.sub(r"^(decoder\.)?norm\.", r"\1output_rms_norm.", name)

        # Embeddings
        name = re.sub(r"embed_tokens\.", r"embeddings.", name)
        name = re.sub(r"lm_head\.", r"output_embeddings.", name)

        out[name] = parameter

    return out
