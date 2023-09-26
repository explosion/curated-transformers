from typing import Any, Callable, Dict, List, Tuple, Union

from ...layers.activations import Activation
from ...util.string import StringRemovePrefix, StringSubInvertible, StringTransform
from ..hf_hub.conversion import process_hf_keys
from .config import GPTNeoXConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    StringSubInvertible((r"gpt_neox", "decoder")),
    # Attention blocks.
    StringSubInvertible((r".attention", ".mha")),
    StringSubInvertible((r".mha.query_key_value", ".mha.input")),
    StringSubInvertible((r".mha.dense", ".mha.output")),
    # Pointwise feedforward.
    StringSubInvertible((r".mlp", ".ffn")),
    StringSubInvertible((r".dense_h_to_4h", ".intermediate")),
    StringSubInvertible((r".ffn.dense_4h_to_h", ".ffn.output")),
    # Layer norms.
    StringSubInvertible((r".input_layernorm", ".attn_input_layer_norm")),
    StringSubInvertible((r".post_attention_layernorm", ".ffn_input_layer_norm")),
    StringSubInvertible((r"final_layer_norm.", "output_layer_norm.")),
    # Embeddings.
    StringSubInvertible((r"embed_in.", "embeddings.piece_embeddings.")),
    StringSubInvertible((r"embed_out.", "output_embeddings.")),
]

DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StringRemovePrefix("gpt_neox.", reversible=False)
] + COMMON_HF_PARAM_KEY_TRANSFORMS
CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS = COMMON_HF_PARAM_KEY_TRANSFORMS

HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "hidden_act": ("activation", Activation),
    "hidden_size": "hidden_width",
    "intermediate_size": "intermediate_width",
    "layer_norm_eps": "layer_norm_eps",
    "max_position_embeddings": "n_positions",
    "num_attention_heads": "n_attention_heads",
    "num_hidden_layers": "n_hidden_layers",
    "rotary_emb_base": "rotary_embedding_base",
    "rotary_pct": "rotary_embedding_fraction",
    "vocab_size": "n_pieces",
}


def convert_hf_config(hf_config: Any) -> GPTNeoXConfig:
    kwargs = process_hf_keys(
        "GPT-NeoX", hf_config, HF_CONFIG_KEY_MAPPING, EXTRA_KWARG_KEYS
    )
    return GPTNeoXConfig(
        model_max_length=hf_config["max_position_embeddings"],
        **kwargs,
    )
