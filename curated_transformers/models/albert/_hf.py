import re
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Tuple, Union

from torch import Tensor

from ...layers.activations import Activation
from ..hf_hub import _process_hf_keys
from .config import ALBERTConfig

HF_KEY_TO_CURATED_KEY = MappingProxyType(
    {
        "embeddings.word_embeddings.weight": "embeddings.piece_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.type_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.embed_output_layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.embed_output_layer_norm.bias",
        # Embedding projection
        "encoder.embedding_hidden_mapping_in.weight": "embeddings.projection.weight",
        "encoder.embedding_hidden_mapping_in.bias": "embeddings.projection.bias",
    }
)

HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "attention_probs_dropout_prob": "attention_probs_dropout_prob",
    "embedding_size": "embedding_width",
    "hidden_act": ("activation", Activation),
    "hidden_dropout_prob": "hidden_dropout_prob",
    "hidden_size": "hidden_width",
    "inner_group_num": "n_layers_per_group",
    "intermediate_size": "intermediate_width",
    "layer_norm_eps": "layer_norm_eps",
    "max_position_embeddings": "n_positions",
    "num_attention_heads": "n_attention_heads",
    "num_hidden_groups": "n_hidden_groups",
    "num_hidden_layers": "n_hidden_layers",
    "type_vocab_size": "n_types",
    "vocab_size": "n_pieces",
}


def convert_hf_config(hf_config: Any) -> ALBERTConfig:
    kwargs = _process_hf_keys("ALBERT", hf_config, HF_CONFIG_KEY_MAPPING)
    return ALBERTConfig(model_max_length=hf_config["max_position_embeddings"], **kwargs)


def convert_hf_state_dict(params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    # Strip the `albert` prefix from ALBERT model parameters.
    stripped_params = {re.sub(r"^albert\.", "", k): v for k, v in params.items()}

    # The ALBERT encoder parameters have the following form:
    #
    # encoder.albert_layer_groups.{hidden_group}.albert_layers.{inner_layer}.{param_name}
    #
    # hidden_group is in [0, n_hidden_group)
    # inner_layer is in [0, n_layers_per_group)

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
        name = re.sub(r"\.mha\.LayerNorm", r".attn_residual_layer_norm", name)
        name = re.sub(r"\.mha\.dense\.", r".mha.output.", name)

        # Pointwise feed-forward layers.
        name = re.sub(r"\.ffn\.", r".ffn.intermediate.", name)
        name = re.sub(r"\.ffn_output\.", r".ffn.output.", name)
        name = re.sub(
            r"\.full_layer_layer_norm\.",
            r".ffn_residual_layer_norm.",
            name,
        )

        out[name] = parameter

    for hf_name, curated_name in HF_KEY_TO_CURATED_KEY.items():
        if hf_name in stripped_params:
            out[curated_name] = stripped_params[hf_name]

    return out
