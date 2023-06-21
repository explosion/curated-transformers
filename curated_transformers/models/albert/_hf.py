import re
from types import MappingProxyType
from typing import Any, Mapping

from torch import Tensor

from .config import AlbertConfig

HF_KEY_TO_CURATED_KEY = MappingProxyType(
    {
        "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
        # Embedding projection
        "encoder.embedding_hidden_mapping_in.weight": "embeddings.projection.weight",
        "encoder.embedding_hidden_mapping_in.bias": "embeddings.projection.bias",
    }
)

EXPECTED_CONFIG_KEYS = {
    "pad_token_id",
    "attention_probs_dropout_prob",
    "embedding_size",
    "hidden_act",
    "hidden_dropout_prob",
    "hidden_size",
    "inner_group_num",
    "intermediate_size",
    "layer_norm_eps",
    "max_position_embeddings",
    "num_attention_heads",
    "num_hidden_groups",
    "num_hidden_layers",
    "type_vocab_size",
    "vocab_size",
}


def convert_hf_config(hf_config: Any) -> AlbertConfig:
    missing_keys = tuple(sorted(EXPECTED_CONFIG_KEYS.difference(set(hf_config.keys()))))
    if len(missing_keys) != 0:
        raise ValueError(f"Missing keys in HF AlBERT model config: {missing_keys}")

    padding_id = hf_config["pad_token_id"]
    return AlbertConfig(
        attention_probs_dropout_prob=hf_config["attention_probs_dropout_prob"],
        embedding_width=hf_config["embedding_size"],
        hidden_act=hf_config["hidden_act"],
        hidden_dropout_prob=hf_config["hidden_dropout_prob"],
        hidden_width=hf_config["hidden_size"],
        inner_group_num=hf_config["inner_group_num"],
        intermediate_width=hf_config["intermediate_size"],
        layer_norm_eps=hf_config["layer_norm_eps"],
        model_max_length=hf_config["max_position_embeddings"],
        max_position_embeddings=hf_config["max_position_embeddings"],
        num_attention_heads=hf_config["num_attention_heads"],
        num_hidden_groups=hf_config["num_hidden_groups"],
        num_hidden_layers=hf_config["num_hidden_layers"],
        padding_id=padding_id,
        type_vocab_size=hf_config["type_vocab_size"],
        vocab_size=hf_config["vocab_size"],
    )


def convert_hf_state_dict(params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    # Strip the `albert` prefix from ALBERT model parameters.
    stripped_params = {re.sub(r"^albert\.", "", k): v for k, v in params.items()}

    # The ALBERT encoder parameters have the following form:
    #
    # encoder.albert_layer_groups.{hidden_group}.albert_layers.{inner_layer}.{param_name}
    #
    # hidden_group is in [0, num_hidden_group)
    # inner_layer is in [0, inner_group_num)

    out = {}
    for name, parameter in stripped_params.items():
        if "encoder.albert_layer" not in name:
            continue

        # TODO: Make these substitutions less ugly.

        # Remove the prefix and rename.
        name = re.sub(r"^encoder\.", "", name)

        # Layer groups
        name = re.sub(r"^albert_layer_groups\.", "groups.", name)

        # Inner layers.
        name = re.sub(r"\.albert_layers\.", ".group_layers.", name)

        # Attention blocks.
        name = re.sub(r"\.attention\.", ".mha.", name)
        name = re.sub(r"\.mha\.LayerNorm", r".attn_output_layernorm", name)
        name = re.sub(r"\.mha\.dense\.", r".mha.output.", name)

        # Pointwise feed-forward layers.
        name = re.sub(r"\.ffn\.", r".ffn.intermediate.", name)
        name = re.sub(r"\.ffn_output\.", r".ffn.output.", name)
        name = re.sub(
            r"\.full_layer_layer_norm\.",
            r".ffn_output_layernorm.",
            name,
        )

        out[name] = parameter

    for hf_name, curated_name in HF_KEY_TO_CURATED_KEY.items():
        if hf_name in stripped_params:
            out[curated_name] = stripped_params[hf_name]

    return out
