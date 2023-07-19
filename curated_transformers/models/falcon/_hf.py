import re
from typing import Any, Dict, List, Mapping

from torch import Tensor

from ..module import DecoderModule
from .config import FalconConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

HF_CONFIG_KEY_MAPPING_REFINED_WEB_MODEL = {
    "hidden_size": "hidden_width",
    "layer_norm_epsilon": "layer_norm_eps",
    "n_head": "num_query_heads",
    "n_layer": "num_hidden_layers",
    "parallel_attn": "parallel_attention",
    "bias": "use_bias",
    "vocab_size": "vocab_size",
}

HF_CONFIG_KEY_MAPPING_FALCON = {
    "hidden_size": "hidden_width",
    "layer_norm_epsilon": "layer_norm_eps",
    "num_attention_heads": "num_query_heads",
    "num_hidden_layers": "num_hidden_layers",
    "parallel_attn": "parallel_attention",
    "bias": "use_bias",
    "vocab_size": "vocab_size",
}


def convert_hf_config(hf_config: Any) -> FalconConfig:
    model_type = hf_config["model_type"]
    if "RefinedWeb" in model_type:
        return _convert_hf_config_refined_web_model(hf_config)
    elif model_type == "falcon":
        return _convert_hf_config_falcon(hf_config)
    else:
        raise ValueError(f"Unknown type of Falcon model: {model_type}")


def _convert_hf_config_refined_web_model(hf_config: Any) -> FalconConfig:
    kwargs = _convert_with_mapping(
        HF_CONFIG_KEY_MAPPING_REFINED_WEB_MODEL, EXTRA_KWARG_KEYS, hf_config
    )

    # For the old configuration format, we can only figure out whether to use
    # the new decoder architecture by checking if `n_head_kv` is present.
    new_decoder_architecture = "n_head_kv" in hf_config
    kwargs["new_decoder_architecture"] = new_decoder_architecture

    if new_decoder_architecture:
        if "n_head_kv" in hf_config:
            kwargs["num_kv_heads"] = hf_config["n_head_kv"]
        else:
            raise ValueError(
                f"Hugging Face Falcon config with new decoder architecture must contain `h_head_kv`"
            )
    else:
        kwargs["new_decoder_architecture"] = False
        if hf_config.get("multi_query", False):
            kwargs["num_kv_heads"] = 1
        else:
            kwargs["num_kv_heads"] = kwargs["num_query_heads"]

    if "alibi" in hf_config and hf_config["alibi"]:
        raise ValueError("Falcon models with ALiBi are currently not supported")

    return FalconConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )


def _convert_hf_config_falcon(hf_config: Any) -> FalconConfig:
    kwargs = _convert_with_mapping(
        HF_CONFIG_KEY_MAPPING_FALCON, EXTRA_KWARG_KEYS, hf_config
    )

    new_decoder_architecture = hf_config.get("new_decoder_architecture", False)
    kwargs["new_decoder_architecture"] = new_decoder_architecture

    if new_decoder_architecture:
        if "num_kv_heads" in hf_config:
            kwargs["num_kv_heads"] = hf_config["num_kv_heads"]
        else:
            raise ValueError(
                f"Hugging Face Falcon config with new decoder architecture must contain `num_kv_heads`"
            )
    else:
        if hf_config.get("multi_query", False):
            kwargs["num_kv_heads"] = 1
        else:
            kwargs["num_kv_heads"] = kwargs["num_query_heads"]

    if "alibi" in hf_config and hf_config["alibi"]:
        raise ValueError("Falcon models with ALiBi are currently not supported")

    return FalconConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )


def _convert_with_mapping(
    hf_config_key_mapping: Dict[str, str],
    opt_keys: List[str],
    hf_config: Dict[str, Any],
) -> Dict[str, Any]:
    hf_config_keys = set(hf_config.keys())
    missing_keys = tuple(
        sorted(set(hf_config_key_mapping.keys()).difference(hf_config_keys))
    )
    if len(missing_keys) != 0:
        raise ValueError(f"Missing keys in Hugging Face Falcon config: {missing_keys}")

    kwargs = {curated: hf_config[hf] for hf, curated in hf_config_key_mapping.items()}
    # Handle config options that are not set in all models.
    kwargs.update({k: hf_config[k] for k in opt_keys if k in hf_config})

    return kwargs


def convert_hf_state_dict(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Convert state dict from HF paramater naming to ours.
    The function is insensitive to prefixes, to allow loading
    both the decoder and the full LM.
    """
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
        name = re.sub(r"\.input_layernorm", r".attn_layer_norm", name)
        name = re.sub(r"\.ln_attn", r".attn_input_layer_norm", name)
        name = re.sub(r"\.post_attention_layernorm", r".ffn_layer_norm", name)
        name = re.sub(r"\.ln_mlp", r".ffn_input_layer_norm", name)
        name = re.sub(r"ln_f\.", r"output_layer_norm.", name)

        # Embeddings
        name = re.sub(r"word_embeddings\.", r"embeddings.", name)
        name = re.sub(r"lm_head\.", r"output_embeddings.", name)

        out[name] = parameter

    return out
