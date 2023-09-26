from typing import Any, Callable, Dict, List, Tuple, Union

from ...layers.activations import Activation
from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import process_hf_keys
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
    kwargs = process_hf_keys("ALBERT", hf_config, HF_CONFIG_KEY_MAPPING)
    return ALBERTConfig(model_max_length=hf_config["max_position_embeddings"], **kwargs)
