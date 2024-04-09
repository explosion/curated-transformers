from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import (
    CommonHFKeys,
    HFConfigKey,
    HFConfigKeyDefault,
    HFSpecificConfig,
    config_from_hf,
    config_to_hf,
)
from ..module import DecoderModule
from .config import GPTNeoXConfig

# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    StringTransformations.sub("gpt_neox", "decoder"),
    # Attention blocks.
    StringTransformations.sub(".attention", ".mha"),
    StringTransformations.sub(".mha.query_key_value", ".mha.input"),
    StringTransformations.sub(".mha.dense", ".mha.output"),
    # Pointwise feedforward.
    StringTransformations.sub(".mlp", ".ffn"),
    StringTransformations.sub(".dense_h_to_4h", ".intermediate"),
    StringTransformations.sub(".ffn.dense_4h_to_h", ".ffn.output"),
    # Layer norms.
    StringTransformations.sub(".input_layernorm", ".attn_input_layer_norm"),
    StringTransformations.sub(".post_attention_layernorm", ".ffn_input_layer_norm"),
    StringTransformations.sub("final_layer_norm.", "output_layer_norm."),
    # Embeddings.
    StringTransformations.sub("embed_in.", "embeddings.piece_embeddings."),
    StringTransformations.sub("embed_out.", "output_embeddings."),
]

DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.remove_prefix("gpt_neox.", reversible=False)
] + COMMON_HF_PARAM_KEY_TRANSFORMS
CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS = COMMON_HF_PARAM_KEY_TRANSFORMS


class HFConfigKeys:
    @staticmethod
    def conv_rotary_embedding_base(config: GPTNeoXConfig) -> int:
        assert config.layer.attention.rotary_embeddings is not None
        return config.layer.attention.rotary_embeddings.rotary_base

    @staticmethod
    def conv_rotary_embedding_fraction(config: GPTNeoXConfig) -> float:
        assert config.layer.attention.rotary_embeddings is not None
        return config.layer.attention.rotary_embeddings.rotary_fraction

    ROTARY_EMB_BASE = HFConfigKey(
        "rotary_emb_base",
        "rotary_embedding_base",
        lambda c: HFConfigKeys.conv_rotary_embedding_base(c),
    )
    ROTARY_PCT = HFConfigKey(
        "rotary_pct",
        "rotary_embedding_fraction",
        lambda c: HFConfigKeys.conv_rotary_embedding_fraction(c),
    )


HF_CONFIG_KEYS: List[Tuple[HFConfigKey, Optional[HFConfigKeyDefault]]] = [
    (CommonHFKeys.DTYPE, HFConfigKeyDefault("float16")),
    (CommonHFKeys.HIDDEN_ACT, None),
    (CommonHFKeys.HIDDEN_SIZE, None),
    (CommonHFKeys.INTERMEDIATE_SIZE, None),
    (CommonHFKeys.LAYER_NORM_EPS, None),
    (CommonHFKeys.VOCAB_SIZE, None),
    (CommonHFKeys.MAX_POSITION_EMBEDDINGS, None),
    (CommonHFKeys.NUM_HIDDEN_LAYERS, None),
    (CommonHFKeys.NUM_ATTENTION_HEADS_UNIFORM, None),
    (HFConfigKeys.ROTARY_EMB_BASE, None),
    (HFConfigKeys.ROTARY_PCT, None),
    (CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
    (CommonHFKeys.HIDDEN_DROPOUT_PROB, HFConfigKeyDefault(0.0)),
]

HF_SPECIFIC_CONFIG_DECODER = HFSpecificConfig(
    architectures=["GPTNeoXModel"], model_type="gpt_neox"
)
HF_SPECIFIC_CONFIG_CAUSAL_LM = HFSpecificConfig(
    architectures=["GPTNeoXForCausalLM"], model_type="gpt_neox"
)


def _config_from_hf(hf_config: Mapping[str, Any]) -> GPTNeoXConfig:
    kwargs = config_from_hf("GPT-NeoX", hf_config, HF_CONFIG_KEYS)

    return GPTNeoXConfig(
        model_max_length=CommonHFKeys.MAX_POSITION_EMBEDDINGS.get_kwarg(kwargs),
        **kwargs,
    )


def _config_to_hf(cls, curated_config: GPTNeoXConfig) -> Dict[str, Any]:
    out = config_to_hf(curated_config, [k for k, _ in HF_CONFIG_KEYS])
    if issubclass(cls, DecoderModule):
        return HF_SPECIFIC_CONFIG_DECODER.merge(out)
    else:
        return HF_SPECIFIC_CONFIG_CAUSAL_LM.merge(out)
