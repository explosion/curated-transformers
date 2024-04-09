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
from .config import MPTConfig

# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    StringTransformations.sub("transformer", "decoder"),
    StringTransformations.sub("blocks", "layers"),
    # Attention blocks.
    StringTransformations.sub(".attn", ".mha"),
    StringTransformations.sub(".Wqkv", ".input"),
    StringTransformations.sub(".out_proj", ".output"),
    # Pointwise feedforward.
    StringTransformations.sub(".up_proj", ".intermediate"),
    StringTransformations.sub("ffn.down_proj", "ffn.output"),
    # Layer norms.
    StringTransformations.sub(".norm_1", ".attn_input_layer_norm"),
    StringTransformations.sub(".norm_2", ".ffn_input_layer_norm"),
    StringTransformations.sub("norm_f.", "output_layer_norm."),
    # Embeddings.
    StringTransformations.sub("wte.", "embeddings.piece_embeddings."),
]

DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.remove_prefix("transformer.", reversible=False)
] + COMMON_HF_PARAM_KEY_TRANSFORMS
CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS = COMMON_HF_PARAM_KEY_TRANSFORMS


class HFConfigKeys:
    @staticmethod
    def conv_intermediate_width_multiplier(config: MPTConfig) -> float:
        return (
            config.layer.feedforward.intermediate_width
            / config.layer.feedforward.hidden_width
        )

    @staticmethod
    def conv_model_max_length(config: MPTConfig) -> int:
        return config.model_max_length

    @staticmethod
    def conv_use_bias(config: MPTConfig) -> int:
        # This needs to be flipped due to how it's
        # stored in the HF config.
        return not config.layer.feedforward.use_bias

    D_MODEL = HFConfigKey(
        "d_model",
        "hidden_width",
        lambda c: CommonCuratedToHFConverters.hidden_width(c),
    )
    EXPANSION_RATIO = HFConfigKey(
        "expansion_ratio",
        "intermediate_width_multiplier",
        lambda c: HFConfigKeys.conv_intermediate_width_multiplier(c),
    )
    MAX_SEQ_LEN = HFConfigKey(
        "max_seq_len",
        "model_max_length",
        lambda c: HFConfigKeys.conv_model_max_length(c),
    )
    N_LAYERS = HFConfigKey(
        "n_layers",
        "n_hidden_layers",
        lambda c: CommonCuratedToHFConverters.n_hidden_layers(c),
    )
    N_HEADS = HFConfigKey(
        "n_heads",
        "n_attention_heads",
        lambda c: CommonCuratedToHFConverters.n_attention_heads_uniform(c),
    )
    NO_BIAS = HFConfigKey(
        "no_bias",
        ("use_bias", lambda v: not v),
        lambda c: HFConfigKeys.conv_use_bias(c),
    )
    LAYER_NORM_EPSILON = HFConfigKey(
        "layer_norm_epsilon",
        "layer_norm_eps",
        lambda c: CommonCuratedToHFConverters.layer_norm_eps(c),
    )


HF_CONFIG_KEYS: List[Tuple[HFConfigKey, Optional[HFConfigKeyDefault]]] = [
    (CommonHFKeys.DTYPE, HFConfigKeyDefault("bfloat16")),
    (HFConfigKeys.D_MODEL, None),
    (HFConfigKeys.EXPANSION_RATIO, None),
    (HFConfigKeys.MAX_SEQ_LEN, None),
    (HFConfigKeys.N_LAYERS, None),
    (HFConfigKeys.N_HEADS, None),
    (HFConfigKeys.NO_BIAS, None),
    (CommonHFKeys.VOCAB_SIZE, None),
    (CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
    (CommonHFKeys.HIDDEN_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
    (HFConfigKeys.LAYER_NORM_EPSILON, HFConfigKeyDefault(1e-5)),
]

HF_SPECIFIC_CONFIG_DECODER = HFSpecificConfig(
    architectures=["MptModel"], model_type="mpt"
)
HF_SPECIFIC_CONFIG_CAUSAL_LM = HFSpecificConfig(
    architectures=["MptForCausalLM"], model_type="mpt"
)


def _config_from_hf(hf_config: Mapping[str, Any]) -> MPTConfig:
    kwargs = config_from_hf("MPT", hf_config, HF_CONFIG_KEYS)
    return MPTConfig(**kwargs)


def _config_to_hf(cls, curated_config: MPTConfig) -> Dict[str, Any]:
    out = config_to_hf(curated_config, [k for k, _ in HF_CONFIG_KEYS])
    if issubclass(cls, DecoderModule):
        return HF_SPECIFIC_CONFIG_DECODER.merge(out)
    else:
        return HF_SPECIFIC_CONFIG_CAUSAL_LM.merge(out)
