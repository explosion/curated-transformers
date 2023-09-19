from typing import Any, Callable, Dict, List, Tuple, Union

from ...layers.activations import Activation
from ...util.string import StringTransform, StrLStrip, StrRepl, StrSub, StrSubInv
from ..hf_hub.conversion import process_hf_keys
from .config import BERTConfig

# Order-dependent.
HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Old HF parameter names (one-way transforms).
    StrSub((r"\.gamma$", ".weight"), backward=None),
    StrSub((r"\.beta$", ".bias"), backward=None),
    # Prefixes.
    StrLStrip("bert.", reversible=False),
    StrSub(
        (r"^encoder\.(layer\.)", "\\1"),
        (r"^(layer\.)", "encoder.\\1"),
    ),
    # Layers.
    StrSub((r"^layer", "layers"), (r"^layers", "layer")),
    # Attention blocks.
    StrSub(
        (r"\.attention\.self\.(query|key|value)", ".mha.\\1"),
        (r"\.mha\.(query|key|value)", ".attention.self.\\1"),
    ),
    StrSubInv((r".attention.output.dense", ".mha.output")),
    StrSubInv((r".attention.output.LayerNorm", ".attn_residual_layer_norm")),
    # Pointwise feed-forward layers.
    StrSubInv((r".intermediate.dense", ".ffn.intermediate")),
    StrSub(
        (r"(\.\d+)\.output\.LayerNorm", "\\1.ffn_residual_layer_norm"),
        (r"(\.\d+)\.ffn_residual_layer_norm", "\\1.output.LayerNorm"),
    ),
    StrSub(
        (r"(\.\d+)\.output\.dense", "\\1.ffn.output"),
        (r"(\.\d+)\.ffn\.output", "\\1.output.dense"),
    ),
    # Embeddings.
    StrRepl("embeddings.word_embeddings.weight", "embeddings.piece_embeddings.weight"),
    StrRepl(
        "embeddings.token_type_embeddings.weight", "embeddings.type_embeddings.weight"
    ),
    StrRepl(
        "embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight"
    ),
    StrRepl("embeddings.LayerNorm.weight", "embeddings.embed_output_layer_norm.weight"),
    StrRepl("embeddings.LayerNorm.bias", "embeddings.embed_output_layer_norm.bias"),
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
