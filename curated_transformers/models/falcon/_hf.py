from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import (
    CommonCuratedToHFConverters,
    CommonHFKeys,
    HFConfigKey,
    HFConfigKeyDefault,
    HFSpecificConfig,
    config_from_hf,
    config_to_hf,
)
from ..module import DecoderModule
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


class HFConfigKeys:
    @staticmethod
    def conv_multi_query(config: FalconConfig) -> bool:
        return (
            config.layer.attention.n_query_heads
            != config.layer.attention.n_key_value_heads
        )

    @staticmethod
    def conv_n_attention_query_heads(config: FalconConfig) -> int:
        return config.layer.attention.n_query_heads

    @staticmethod
    def conv_n_attention_keyvalue_heads(config: FalconConfig) -> int:
        return config.layer.attention.n_key_value_heads

    @staticmethod
    def conv_use_bias(config: FalconConfig) -> bool:
        return config.layer.feedforward.use_bias

    @staticmethod
    def conv_use_alibi(config: FalconConfig) -> bool:
        return config.layer.attention.use_alibi

    @staticmethod
    def conv_use_parallel_attention(config: FalconConfig) -> bool:
        return config.layer.attention.use_parallel_attention

    @staticmethod
    def conv_new_decoder_architecture(config: FalconConfig) -> bool:
        return config.new_decoder_architecture

    # Used by Refined Web Model.
    N_LAYER = HFConfigKey(
        "n_layer",
        "n_hidden_layers",
        lambda c: CommonCuratedToHFConverters.n_hidden_layers(c),
    )
    N_HEAD = HFConfigKey(
        "n_head",
        "n_query_heads",
        lambda c: HFConfigKeys.conv_n_attention_query_heads(c),
    )
    N_HEAD_KV = HFConfigKey(
        "n_head_kv",
        "n_key_value_heads",
        lambda c: HFConfigKeys.conv_n_attention_keyvalue_heads(c),
    )
    # Used by Falcon.
    NUM_HEAD_KV = HFConfigKey(
        "num_kv_heads",
        "n_key_value_heads",
        lambda c: HFConfigKeys.conv_n_attention_keyvalue_heads(c),
    )
    NUM_ATTENTION_HEADS = HFConfigKey(
        "num_attention_heads",
        "n_query_heads",
        lambda c: HFConfigKeys.conv_n_attention_query_heads(c),
    )
    # Used by both.
    LAYER_NORM_EPSILON = HFConfigKey(
        "layer_norm_epsilon",
        "layer_norm_eps",
        lambda c: CommonCuratedToHFConverters.layer_norm_eps(c),
    )
    BIAS = HFConfigKey(
        "bias",
        "use_bias",
        lambda c: HFConfigKeys.conv_use_bias(c),
    )
    ALIBI = HFConfigKey(
        "alibi",
        "use_alibi",
        lambda c: HFConfigKeys.conv_use_alibi(c),
    )
    PARALLEL_ATTN = HFConfigKey(
        "parallel_attn",
        "use_parallel_attention",
        lambda c: HFConfigKeys.conv_use_parallel_attention(c),
    )
    NEW_DECODER_ARCHITECTURE = HFConfigKey(
        "new_decoder_architecture",
        "new_decoder_architecture",
        lambda c: HFConfigKeys.conv_new_decoder_architecture(c),
    )
    # The following keys are not directly converted to kwargs
    # but still need to serialized back to the HF config.
    MULTI_QUERY = HFConfigKey(
        "multi_query",
        "multi_query",
        lambda c: HFConfigKeys.conv_multi_query(c),
    )


# Corresponds to the implementation that the Falcon models
# originally shipped with.
HF_CONFIG_KEYS_REFINED_WEB_MODEL: List[
    Tuple[HFConfigKey, Optional[HFConfigKeyDefault]]
] = [
    (CommonHFKeys.DTYPE, HFConfigKeyDefault("bfloat16")),
    (CommonHFKeys.HIDDEN_SIZE, None),
    (HFConfigKeys.N_HEAD, None),
    (HFConfigKeys.N_HEAD_KV, HFConfigKeyDefault(-1)),
    (HFConfigKeys.MULTI_QUERY, HFConfigKeyDefault(False)),
    (HFConfigKeys.N_LAYER, None),
    (HFConfigKeys.PARALLEL_ATTN, None),
    (HFConfigKeys.BIAS, None),
    (HFConfigKeys.ALIBI, None),
    (HFConfigKeys.LAYER_NORM_EPSILON, None),
    (CommonHFKeys.VOCAB_SIZE, None),
    (CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
    (CommonHFKeys.HIDDEN_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
]

# Corresponds to the mainline implementation for Falcon models
# in the `transformers` library.
HF_CONFIG_KEYS_FALCON: List[Tuple[HFConfigKey, Optional[HFConfigKeyDefault]]] = [
    (CommonHFKeys.DTYPE, HFConfigKeyDefault("bfloat16")),
    (CommonHFKeys.HIDDEN_SIZE, None),
    (HFConfigKeys.NUM_ATTENTION_HEADS, None),
    (HFConfigKeys.NUM_HEAD_KV, HFConfigKeyDefault(-1)),
    (HFConfigKeys.MULTI_QUERY, HFConfigKeyDefault(False)),
    (CommonHFKeys.NUM_HIDDEN_LAYERS, None),
    (HFConfigKeys.PARALLEL_ATTN, None),
    (HFConfigKeys.BIAS, None),
    (HFConfigKeys.ALIBI, None),
    (HFConfigKeys.LAYER_NORM_EPSILON, None),
    (CommonHFKeys.VOCAB_SIZE, None),
    (HFConfigKeys.NEW_DECODER_ARCHITECTURE, HFConfigKeyDefault(False)),
    (CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
    (CommonHFKeys.HIDDEN_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
]


HF_SPECIFIC_CONFIG_DECODER = HFSpecificConfig(
    architectures=["FalconModel"], model_type="falcon"
)
HF_SPECIFIC_CONFIG_CAUSAL_LM = HFSpecificConfig(
    architectures=["FalconForCausalLM"], model_type="falcon"
)


def _config_from_hf_refined_web_model(hf_config: Mapping[str, Any]) -> FalconConfig:
    kwargs = config_from_hf("Falcon", hf_config, HF_CONFIG_KEYS_REFINED_WEB_MODEL)

    # For the old configuration format, we can only figure out whether to use
    # the new decoder architecture by checking if `n_head_kv` is present.
    new_decoder_architecture = HFConfigKeys.N_HEAD_KV.get_kwarg(kwargs) != -1
    multi_query = HFConfigKeys.MULTI_QUERY.get_kwarg(kwargs)

    HFConfigKeys.NEW_DECODER_ARCHITECTURE.set_kwarg(new_decoder_architecture, kwargs)
    HFConfigKeys.MULTI_QUERY.remove_kwarg(kwargs)

    if not new_decoder_architecture:
        if multi_query:
            HFConfigKeys.N_HEAD_KV.set_kwarg(1, kwargs)
        else:
            HFConfigKeys.N_HEAD_KV.set_kwarg(
                HFConfigKeys.N_HEAD.get_kwarg(kwargs), kwargs
            )

    return FalconConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )


def _config_from_hf_falcon(hf_config: Mapping[str, Any]) -> FalconConfig:
    kwargs = config_from_hf("Falcon", hf_config, HF_CONFIG_KEYS_FALCON)

    new_decoder_architecture = HFConfigKeys.NEW_DECODER_ARCHITECTURE.get_kwarg(kwargs)
    multi_query = HFConfigKeys.MULTI_QUERY.get_kwarg(kwargs)

    HFConfigKeys.MULTI_QUERY.remove_kwarg(kwargs)

    if new_decoder_architecture:
        if HFConfigKeys.NUM_HEAD_KV.get_kwarg(kwargs) == -1:
            raise ValueError(
                f"Hugging Face Falcon config with new decoder architecture must contain `n_head_kv`"
            )
    else:
        if multi_query:
            HFConfigKeys.NUM_HEAD_KV.set_kwarg(1, kwargs)
        else:
            HFConfigKeys.NUM_HEAD_KV.set_kwarg(
                HFConfigKeys.NUM_ATTENTION_HEADS.get_kwarg(kwargs), kwargs
            )

    return FalconConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )


def _config_from_hf(hf_config: Mapping[str, Any]) -> FalconConfig:
    model_type = hf_config["model_type"]
    if "RefinedWeb" in model_type:
        return _config_from_hf_refined_web_model(hf_config)
    elif model_type == "falcon":
        return _config_from_hf_falcon(hf_config)
    else:
        raise ValueError(f"Unknown type of Falcon model: {model_type}")


def _config_to_hf(cls, curated_config: FalconConfig) -> Dict[str, Any]:
    # We only support exporting as the mainline Falcon implementation.
    out = config_to_hf(curated_config, [k for k, _ in HF_CONFIG_KEYS_FALCON])
    if issubclass(cls, DecoderModule):
        return HF_SPECIFIC_CONFIG_DECODER.merge(out)
    else:
        return HF_SPECIFIC_CONFIG_CAUSAL_LM.merge(out)
