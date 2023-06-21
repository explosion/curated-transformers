import re
from typing import Any, Mapping

from torch import Tensor

from ..module import DecoderModule
from .config import RefinedWebModelConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]


HF_CONFIG_KEY_MAPPING = {
    "hidden_size": "hidden_width",
    "layer_norm_epsilon": "layer_norm_eps",
    "multi_query": "multi_query",
    "n_head": "num_attention_heads",
    "n_layer": "num_hidden_layers",
    "bias": "use_bias",
    "vocab_size": "vocab_size",
}


def convert_hf_config(hf_config: Any) -> RefinedWebModelConfig:
    missing_keys = tuple(
        sorted(set(HF_CONFIG_KEY_MAPPING.keys()).difference(set(hf_config.keys())))
    )
    if len(missing_keys) != 0:
        raise ValueError(f"Missing keys in HF RefinedWeb model config: {missing_keys}")

    kwargs = {curated: hf_config[hf] for hf, curated in HF_CONFIG_KEY_MAPPING.items()}
    # Handle config options that are not set in all models.
    kwargs.update({k: hf_config[k] for k in EXTRA_KWARG_KEYS if k in hf_config})

    if "parallel_attn" in hf_config and not hf_config["parallel_attn"]:
        raise ValueError(
            "Refined Web Models without parallel attention are currently not supported"
        )
    if "alibi" in hf_config and hf_config["alibi"]:
        raise ValueError("Refined Web Models with ALiBi are currently not supported")

    return RefinedWebModelConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )


def convert_hf_state_dict(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Convert state dict from HF paramater naming to ours.
    The function is insensitive to prefixes, to allow loading
    both the decoder and the full LM."""
    if issubclass(cls, DecoderModule):
        stripped_params = {
            re.sub(r"^transformer\.", "", k): v for k, v in params.items()
        }
    else:
        stripped_params = {
            re.sub(r"^transformer\.", "decoder.", k): v for k, v in params.items()
        }

    out = {}
    for name, parameter in stripped_params.items():
        # These parameters are all created on-the-fly.
        if "rotary_emb" in name or "attention.bias" in name or "masked_bias" in name:
            continue

        name = re.sub(r"^h\.", "layers.", name)
        name = re.sub(r"decoder\.h\.", "decoder.layers.", name)

        # Attention
        name = re.sub(r"\.self_attention", r".mha", name)
        name = re.sub(r"\.query_key_value", r".input", name)
        name = re.sub(r"\.mha\.dense", r".mha.output", name)

        # Pointwise feedforward
        name = re.sub(r"\.mlp", r".ffn", name)
        name = re.sub(r"\.dense_h_to_4h", r".intermediate", name)
        name = re.sub(r"\.dense_4h_to_h", r".output", name)

        # Layer norms
        name = re.sub(r"\.input_layernorm", r".input_layer_norm", name)
        name = re.sub(r"ln_f\.", r"output_layer_norm.", name)

        # Embeddings
        name = re.sub(r"word_embeddings\.", r"embeddings.", name)
        name = re.sub(r"lm_head\.", r"output_embeddings.", name)

        out[name] = parameter

    return out
