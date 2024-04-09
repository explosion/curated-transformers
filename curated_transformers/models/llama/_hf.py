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
from .config import LlamaConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Attention blocks.
    StringTransformations.sub(".self_attn", ".mha"),
    StringTransformations.sub(".q_proj", ".query"),
    StringTransformations.sub(".k_proj", ".key"),
    StringTransformations.sub(".v_proj", ".value"),
    StringTransformations.sub(".o_proj", ".output"),
    # Pointwise feedforward
    StringTransformations.sub(".mlp", ".ffn"),
    StringTransformations.sub(".up_proj", ".intermediate"),
    StringTransformations.sub("ffn.down_proj", "ffn.output"),
    StringTransformations.sub(".gate_proj", ".gate"),
    # RMS norms
    StringTransformations.sub(".input_layernorm", ".attn_input_layer_norm"),
    StringTransformations.sub(".post_attention_layernorm", ".ffn_input_layer_norm"),
    StringTransformations.regex_sub(
        (r"^(decoder\.)?norm\.", "\\1output_layer_norm."),
        (r"^(decoder\.)?output_layer_norm\.", "\\1norm."),
    ),
    # Embeddings
    StringTransformations.sub("embed_tokens.", "embeddings.piece_embeddings."),
    StringTransformations.sub("lm_head.", "output_embeddings."),
]

DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.remove_prefix("model.", reversible=False)
] + COMMON_HF_PARAM_KEY_TRANSFORMS
CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.sub("model.", "decoder.")
] + COMMON_HF_PARAM_KEY_TRANSFORMS


class HFConfigKeys:
    @staticmethod
    def conv_n_attention_query_heads(config: LlamaConfig) -> float:
        return config.layer.attention.n_query_heads

    @staticmethod
    def conv_n_attention_keyvalue_heads(config: LlamaConfig) -> float:
        return config.layer.attention.n_key_value_heads

    NUM_ATTENTION_HEADS = HFConfigKey(
        "num_attention_heads",
        "n_query_heads",
        lambda c: HFConfigKeys.conv_n_attention_query_heads(c),
    )
    RMS_NORM_EPS = HFConfigKey(
        "rms_norm_eps",
        "rms_norm_eps",
        lambda c: CommonCuratedToHFConverters.layer_norm_eps(c),
    )
    NUM_KEY_VALUE_HEADS = HFConfigKey(
        "num_key_value_heads",
        "n_key_value_heads",
        lambda c: HFConfigKeys.conv_n_attention_keyvalue_heads(c),
    )


HF_CONFIG_KEYS: List[Tuple[HFConfigKey, Optional[HFConfigKeyDefault]]] = [
    (CommonHFKeys.DTYPE, HFConfigKeyDefault("float16")),
    (CommonHFKeys.HIDDEN_ACT, None),
    (CommonHFKeys.HIDDEN_SIZE, None),
    (CommonHFKeys.INTERMEDIATE_SIZE, None),
    (CommonHFKeys.VOCAB_SIZE, None),
    (CommonHFKeys.NUM_HIDDEN_LAYERS, None),
    (HFConfigKeys.NUM_ATTENTION_HEADS, None),
    (HFConfigKeys.RMS_NORM_EPS, None),
    (HFConfigKeys.NUM_KEY_VALUE_HEADS, HFConfigKeyDefault(-1)),
    (CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
    (CommonHFKeys.HIDDEN_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
]

HF_SPECIFIC_CONFIG_DECODER = HFSpecificConfig(
    architectures=["LlamaModel"], model_type="llama"
)
HF_SPECIFIC_CONFIG_CAUSAL_LM = HFSpecificConfig(
    architectures=["LlamaForCausalLM"], model_type="llama"
)


def _config_from_hf(hf_config: Mapping[str, Any]) -> LlamaConfig:
    kwargs = config_from_hf("Llama", hf_config, HF_CONFIG_KEYS)

    n_kv_heads = HFConfigKeys.NUM_KEY_VALUE_HEADS.get_kwarg(kwargs)
    if n_kv_heads == -1:
        HFConfigKeys.NUM_KEY_VALUE_HEADS.set_kwarg(
            HFConfigKeys.NUM_ATTENTION_HEADS, kwargs
        )

    return LlamaConfig(
        rotary_embedding_base=10000, rotary_embedding_fraction=1.0, **kwargs
    )


def _config_to_hf(cls, curated_config: LlamaConfig) -> Dict[str, Any]:
    out = config_to_hf(curated_config, [k for k, _ in HF_CONFIG_KEYS])
    if issubclass(cls, DecoderModule):
        return HF_SPECIFIC_CONFIG_DECODER.merge(out)
    else:
        return HF_SPECIFIC_CONFIG_CAUSAL_LM.merge(out)
