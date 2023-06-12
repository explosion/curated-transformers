from typing import Any, List, Mapping
import itertools
import re
import torch
from torch import Tensor


from .config import AlbertConfig
from ..util.serde import DeserializationParamBucket, RegExParameterBucket


def convert_hf_config(hf_config: Any) -> AlbertConfig:
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

    # Rename and move embedding parameters to the inner BertEmbeddings module.
    key_map = {
        "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
        # Embedding projection
        "encoder.embedding_hidden_mapping_in.weight": "embeddings.projection.weight",
        "encoder.embedding_hidden_mapping_in.bias": "embeddings.projection.bias",
    }

    for hf_name, curated_name in key_map.items():
        if hf_name in stripped_params:
            out[curated_name] = stripped_params[hf_name]

    return _merge_qkv_albert(out)


def deserialization_param_buckets(
    num_groups: int, num_layers_per_group: int
) -> List[DeserializationParamBucket]:
    out = []
    for group, layer in itertools.product(
        range(num_groups), range(num_layers_per_group)
    ):
        regex_str = rf"groups\.{group}\.group_layers\.{layer}\.mha\.(query|key|value).(weight|bias)"
        expected_keys = {
            rf"groups.{group}.group_layers.{layer}.mha.{module}.{param}"
            for module, param in itertools.product(
                ["query", "key", "value"], ["weight", "bias"]
            )
        }
        out.append(RegExParameterBucket(pattern=regex_str, expected_keys=expected_keys))
    return out  # type: ignore


def _merge_qkv_albert(params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    out = {}
    for name, parameter in params.items():
        m = re.match(
            r"groups\.(?P<group>[0-9]+)\.group_layers\.(?P<layer>[0-9]+)\.mha\.(query|key|value).(?P<param_type>weight|bias)",
            name,
        )
        if m:
            if "query" in name:
                base = f"groups.{m['group']}.group_layers.{m['layer']}.mha"
                out[f"{base}.input.{m['param_type']}"] = torch.cat(
                    [
                        parameter,
                        params[f"{base}.key.{m['param_type']}"],
                        params[f"{base}.value.{m['param_type']}"],
                    ]
                )
            continue
        out[name] = parameter

    return out
