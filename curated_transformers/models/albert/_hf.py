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
from .config import ALBERTConfig

# Order-dependent.
HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Prefixes.
    StringTransformations.remove_prefix("albert.", reversible=False),
    StringTransformations.regex_sub(
        (r"^encoder\.(embedding_|albert_layer)", "\\1"),
        (r"^(embedding_|albert_layer)", "encoder.\\1"),
    ),
    # Layer groups
    StringTransformations.regex_sub(
        (r"^albert_layer_groups\.", "groups."), (r"^groups\.", "albert_layer_groups.")
    ),
    # Inner layers.
    StringTransformations.sub(".albert_layers.", ".group_layers."),
    # Attention blocks.
    StringTransformations.sub(".attention.", ".mha."),
    StringTransformations.sub(".mha.LayerNorm", ".attn_residual_layer_norm"),
    StringTransformations.sub(".mha.dense", ".mha.output"),
    # Pointwise feed-forward layers.
    StringTransformations.sub(".ffn.", ".ffn.intermediate."),
    StringTransformations.sub(".ffn_output.", ".ffn.output."),
    StringTransformations.sub(".full_layer_layer_norm.", ".ffn_residual_layer_norm."),
    # Embeddings.
    StringTransformations.replace(
        "embeddings.word_embeddings.weight", "embeddings.piece_embeddings.weight"
    ),
    StringTransformations.replace(
        "embeddings.token_type_embeddings.weight", "embeddings.type_embeddings.weight"
    ),
    StringTransformations.replace(
        "embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight"
    ),
    StringTransformations.replace(
        "embeddings.LayerNorm.weight", "embeddings.embed_output_layer_norm.weight"
    ),
    StringTransformations.replace(
        "embeddings.LayerNorm.bias", "embeddings.embed_output_layer_norm.bias"
    ),
    # Embedding projection.
    StringTransformations.replace(
        "embedding_hidden_mapping_in.weight", "embeddings.projection.weight"
    ),
    StringTransformations.replace(
        "embedding_hidden_mapping_in.bias", "embeddings.projection.bias"
    ),
]


class HFConfigKeys:
    @staticmethod
    def conv_n_layers_per_group(config: ALBERTConfig) -> int:
        return config.layer.n_layers_per_group

    @staticmethod
    def conv_n_hidden_groups(config: ALBERTConfig) -> int:
        return config.layer.n_hidden_groups

    INNER_GROUP_NUM = HFConfigKey(
        "inner_group_num",
        "n_layers_per_group",
        lambda c: HFConfigKeys.conv_n_layers_per_group(c),
    )
    NUM_HIDDEN_GROUPS = HFConfigKey(
        "num_hidden_groups",
        "n_hidden_groups",
        lambda c: HFConfigKeys.conv_n_hidden_groups(c),
    )


HF_CONFIG_KEYS: List[Tuple[HFConfigKey, Optional[HFConfigKeyDefault]]] = [
    (CommonHFKeys.ATTENTION_PROBS_DROPOUT_PROB, None),
    (CommonHFKeys.DTYPE, HFConfigKeyDefault("float32")),
    (CommonHFKeys.EMBEDDING_SIZE, None),
    (CommonHFKeys.HIDDEN_DROPOUT_PROB, None),
    (CommonHFKeys.HIDDEN_SIZE, None),
    (CommonHFKeys.HIDDEN_ACT, None),
    (CommonHFKeys.INTERMEDIATE_SIZE, None),
    (CommonHFKeys.LAYER_NORM_EPS, None),
    (CommonHFKeys.NUM_ATTENTION_HEADS_UNIFORM, None),
    (CommonHFKeys.NUM_HIDDEN_LAYERS, None),
    (CommonHFKeys.VOCAB_SIZE, None),
    (CommonHFKeys.TYPE_VOCAB_SIZE, None),
    (CommonHFKeys.MAX_POSITION_EMBEDDINGS, None),
    (HFConfigKeys.INNER_GROUP_NUM, None),
    (HFConfigKeys.NUM_HIDDEN_GROUPS, None),
]

HF_SPECIFIC_CONFIG = HFSpecificConfig(
    architectures=["AlbertModel"], model_type="albert"
)


def _config_from_hf(hf_config: Mapping[str, Any]) -> ALBERTConfig:
    kwargs = config_from_hf("ALBERT", hf_config, HF_CONFIG_KEYS)
    return ALBERTConfig(
        model_max_length=CommonHFKeys.MAX_POSITION_EMBEDDINGS.get_kwarg(kwargs),
        **kwargs
    )


def _config_to_hf(curated_config: ALBERTConfig) -> Dict[str, Any]:
    out = config_to_hf(curated_config, [k for k, _ in HF_CONFIG_KEYS])
    return HF_SPECIFIC_CONFIG.merge(out)
