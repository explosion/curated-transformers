from typing import Any, Callable, Dict, List, Tuple, Union

from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import process_hf_keys
from .config import FalconConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]


# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    StringTransformations.regex_sub((r"^h\.", "layers."), (r"^layers\.", "h.")),
    StringTransformations.sub("decoder.h.", "decoder.layers."),
    # Attention blocks.
    StringTransformations.sub(".self_attention", ".mha"),
    StringTransformations.sub(".mha.query_key_value", ".mha.input"),
    StringTransformations.sub(".mha.dense", ".mha.output"),
    # Pointwise feedforward.
    StringTransformations.sub(".mlp", ".ffn"),
    StringTransformations.sub(".dense_h_to_4h", ".intermediate"),
    StringTransformations.sub(".ffn.dense_4h_to_h", ".ffn.output"),
    # Layer norms.
    StringTransformations.sub(".input_layernorm", ".attn_layer_norm"),
    StringTransformations.sub(".ln_attn", ".attn_input_layer_norm"),
    StringTransformations.sub(".post_attention_layernorm", ".ffn_layer_norm"),
    StringTransformations.sub(".ln_mlp", ".ffn_input_layer_norm"),
    StringTransformations.sub("ln_f.", "output_layer_norm."),
    # Embeddings.
    StringTransformations.sub("word_embeddings.", "embeddings.piece_embeddings."),
    StringTransformations.sub("lm_head.", "output_embeddings."),
]

DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.remove_prefix("transformer.", reversible=False)
] + COMMON_HF_PARAM_KEY_TRANSFORMS
CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.sub("transformer.", "decoder."),
] + COMMON_HF_PARAM_KEY_TRANSFORMS

HF_CONFIG_KEY_MAPPING_REFINED_WEB_MODEL: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "hidden_size": "hidden_width",
    "layer_norm_epsilon": "layer_norm_eps",
    "n_head": "n_query_heads",
    "n_layer": "n_hidden_layers",
    "parallel_attn": "use_parallel_attention",
    "bias": "use_bias",
    "vocab_size": "n_pieces",
    "alibi": "use_alibi",
}

HF_CONFIG_KEY_MAPPING_FALCON: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "hidden_size": "hidden_width",
    "layer_norm_epsilon": "layer_norm_eps",
    "num_attention_heads": "n_query_heads",
    "num_hidden_layers": "n_hidden_layers",
    "parallel_attn": "use_parallel_attention",
    "bias": "use_bias",
    "vocab_size": "n_pieces",
    "alibi": "use_alibi",
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
    kwargs = process_hf_keys(
        "Falcon", hf_config, HF_CONFIG_KEY_MAPPING_REFINED_WEB_MODEL, EXTRA_KWARG_KEYS
    )

    # For the old configuration format, we can only figure out whether to use
    # the new decoder architecture by checking if `n_head_kv` is present.
    new_decoder_architecture = "n_head_kv" in hf_config
    kwargs["new_decoder_architecture"] = new_decoder_architecture

    if new_decoder_architecture:
        if "n_head_kv" in hf_config:
            kwargs["n_key_value_heads"] = hf_config["n_head_kv"]
        else:
            raise ValueError(
                f"Hugging Face Falcon config with new decoder architecture must contain `n_head_kv`"
            )
    else:
        kwargs["new_decoder_architecture"] = False
        if hf_config.get("multi_query", False):
            kwargs["n_key_value_heads"] = 1
        else:
            kwargs["n_key_value_heads"] = kwargs["n_query_heads"]

    if "alibi" in hf_config and hf_config["alibi"]:
        raise ValueError("Falcon models with ALiBi are currently not supported")

    return FalconConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )


def _convert_hf_config_falcon(hf_config: Any) -> FalconConfig:
    kwargs = process_hf_keys(
        "Falcon", hf_config, HF_CONFIG_KEY_MAPPING_FALCON, EXTRA_KWARG_KEYS
    )

    new_decoder_architecture = hf_config.get("new_decoder_architecture", False)
    kwargs["new_decoder_architecture"] = new_decoder_architecture

    if new_decoder_architecture:
        if "num_kv_heads" in hf_config:
            kwargs["n_key_value_heads"] = hf_config["num_kv_heads"]
        else:
            raise ValueError(
                f"Hugging Face Falcon config with new decoder architecture must contain `num_kv_heads`"
            )
    else:
        if hf_config.get("multi_query", False):
            kwargs["n_key_value_heads"] = 1
        else:
            kwargs["n_key_value_heads"] = kwargs["n_query_heads"]

    return FalconConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )
