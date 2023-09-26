from typing import Any, Callable, Dict, List, Tuple, Union

from ...layers.activations import Activation
from ...util.string import (
    StringRemovePrefix,
    StringReplace,
    StringSubInvertible,
    StringSubRegEx,
    StringTransform,
)
from ..hf_hub.conversion import process_hf_keys
from .config import ALBERTConfig

# Order-dependent.
HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Prefixes.
    StringRemovePrefix("albert.", reversible=False),
    StringSubRegEx(
        (r"^encoder\.(embedding_|albert_layer)", "\\1"),
        (r"^(embedding_|albert_layer)", "encoder.\\1"),
    ),
    # Layer groups
    StringSubRegEx(
        (r"^albert_layer_groups\.", "groups."), (r"^groups\.", "albert_layer_groups.")
    ),
    # Inner layers.
    StringSubInvertible((".albert_layers.", ".group_layers.")),
    # Attention blocks.
    StringSubInvertible((".attention.", ".mha.")),
    StringSubInvertible((".mha.LayerNorm", ".attn_residual_layer_norm")),
    StringSubInvertible((".mha.dense", ".mha.output")),
    # Pointwise feed-forward layers.
    StringSubInvertible((".ffn.", ".ffn.intermediate.")),
    StringSubInvertible((".ffn_output.", ".ffn.output.")),
    StringSubInvertible((".full_layer_layer_norm.", ".ffn_residual_layer_norm.")),
    # Embeddings.
    StringReplace(
        "embeddings.word_embeddings.weight", "embeddings.piece_embeddings.weight"
    ),
    StringReplace(
        "embeddings.token_type_embeddings.weight", "embeddings.type_embeddings.weight"
    ),
    StringReplace(
        "embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight"
    ),
    StringReplace(
        "embeddings.LayerNorm.weight", "embeddings.embed_output_layer_norm.weight"
    ),
    StringReplace(
        "embeddings.LayerNorm.bias", "embeddings.embed_output_layer_norm.bias"
    ),
    # Embedding projection.
    StringReplace("embedding_hidden_mapping_in.weight", "embeddings.projection.weight"),
    StringReplace("embedding_hidden_mapping_in.bias", "embeddings.projection.bias"),
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
