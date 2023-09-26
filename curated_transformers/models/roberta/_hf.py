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
from .config import RoBERTaConfig

# Order-dependent.
HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Prefixes.
    StringRemovePrefix("roberta.", reversible=False),
    StringSubRegEx(
        (r"^encoder\.(layer\.)", "\\1"),
        (r"^(layer\.)", "encoder.\\1"),
    ),
    # Layers.
    StringSubRegEx((r"^layer", "layers"), (r"^layers", "layer")),
    # Attention blocks.
    StringSubRegEx(
        (r"\.attention\.self\.(query|key|value)", ".mha.\\1"),
        (r"\.mha\.(query|key|value)", ".attention.self.\\1"),
    ),
    StringSubInvertible((r".attention.output.dense", ".mha.output")),
    StringSubInvertible((r".attention.output.LayerNorm", ".attn_residual_layer_norm")),
    # Pointwise feed-forward layers.
    StringSubInvertible((r".intermediate.dense", ".ffn.intermediate")),
    StringSubRegEx(
        (r"(\.\d+)\.output\.LayerNorm", "\\1.ffn_residual_layer_norm"),
        (r"(\.\d+)\.ffn_residual_layer_norm", "\\1.output.LayerNorm"),
    ),
    StringSubRegEx(
        (r"(\.\d+)\.output\.dense", "\\1.ffn.output"),
        (r"(\.\d+)\.ffn\.output", "\\1.output.dense"),
    ),
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
]

HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "pad_token_id": "padding_id",
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


def convert_hf_config(hf_config: Any) -> RoBERTaConfig:
    kwargs = process_hf_keys("RoBERTa", hf_config, HF_CONFIG_KEY_MAPPING)

    return RoBERTaConfig(
        embedding_width=hf_config["hidden_size"],
        # Positions embeddings for 0..padding_id are reserved.
        model_max_length=hf_config["max_position_embeddings"]
        - (kwargs["padding_id"] + 1),
        **kwargs,
    )
