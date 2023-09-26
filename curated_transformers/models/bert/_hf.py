from typing import Any, Callable, Dict, List, Tuple, Union

from ...layers.activations import Activation
from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import process_hf_keys
from .config import BERTConfig

# Order-dependent.
HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Old HF parameter names (one-way transforms).
    StringTransformations.regex_sub((r"\.gamma$", ".weight"), backward=None),
    StringTransformations.regex_sub((r"\.beta$", ".bias"), backward=None),
    # Prefixes.
    StringTransformations.remove_prefix("bert.", reversible=False),
    StringTransformations.regex_sub(
        (r"^encoder\.(layer\.)", "\\1"),
        (r"^(layer\.)", "encoder.\\1"),
    ),
    # Layers.
    StringTransformations.regex_sub((r"^layer", "layers"), (r"^layers", "layer")),
    # Attention blocks.
    StringTransformations.regex_sub(
        (r"\.attention\.self\.(query|key|value)", ".mha.\\1"),
        (r"\.mha\.(query|key|value)", ".attention.self.\\1"),
    ),
    StringTransformations.sub(".attention.output.dense", ".mha.output"),
    StringTransformations.sub(
        r".attention.output.LayerNorm", ".attn_residual_layer_norm"
    ),
    # Pointwise feed-forward layers.
    StringTransformations.sub(".intermediate.dense", ".ffn.intermediate"),
    StringTransformations.regex_sub(
        (r"(\.\d+)\.output\.LayerNorm", "\\1.ffn_residual_layer_norm"),
        (r"(\.\d+)\.ffn_residual_layer_norm", "\\1.output.LayerNorm"),
    ),
    StringTransformations.regex_sub(
        (r"(\.\d+)\.output\.dense", "\\1.ffn.output"),
        (r"(\.\d+)\.ffn\.output", "\\1.output.dense"),
    ),
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
]

HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "attention_probs_dropout_prob": "attention_probs_dropout_prob",
    "hidden_act": ("activation", Activation),
    "hidden_dropout_prob": "hidden_dropout_prob",
    "hidden_size": "hidden_width",
    "intermediate_size": "intermediate_width",
    "layer_norm_eps": "layer_norm_eps",
    "max_position_embeddings": "n_positions",
    "num_attention_heads": "n_attention_heads",
    "num_hidden_layers": "n_hidden_layers",
    "type_vocab_size": "n_types",
    "vocab_size": "n_pieces",
}


def convert_hf_config(hf_config: Any) -> BERTConfig:
    kwargs = process_hf_keys("BERT", hf_config, HF_CONFIG_KEY_MAPPING)

    return BERTConfig(
        embedding_width=hf_config["hidden_size"],
        model_max_length=hf_config["max_position_embeddings"],
        **kwargs,
    )
