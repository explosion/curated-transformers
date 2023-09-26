from typing import Any, Callable, Dict, List, Tuple, Union

from ...layers.activations import Activation
from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import process_hf_keys
from .config import GPTNeoXConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    StringTransformations.regex_sub_invertible((r"gpt_neox", "decoder")),
    # Attention blocks.
    StringTransformations.regex_sub_invertible((r".attention", ".mha")),
    StringTransformations.regex_sub_invertible((r".mha.query_key_value", ".mha.input")),
    StringTransformations.regex_sub_invertible((r".mha.dense", ".mha.output")),
    # Pointwise feedforward.
    StringTransformations.regex_sub_invertible((r".mlp", ".ffn")),
    StringTransformations.regex_sub_invertible((r".dense_h_to_4h", ".intermediate")),
    StringTransformations.regex_sub_invertible((r".ffn.dense_4h_to_h", ".ffn.output")),
    # Layer norms.
    StringTransformations.regex_sub_invertible(
        (r".input_layernorm", ".attn_input_layer_norm")
    ),
    StringTransformations.regex_sub_invertible(
        (r".post_attention_layernorm", ".ffn_input_layer_norm")
    ),
    StringTransformations.regex_sub_invertible(
        (r"final_layer_norm.", "output_layer_norm.")
    ),
    # Embeddings.
    StringTransformations.regex_sub_invertible(
        (r"embed_in.", "embeddings.piece_embeddings.")
    ),
    StringTransformations.regex_sub_invertible((r"embed_out.", "output_embeddings.")),
]

DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.remove_prefix("gpt_neox.", reversible=False)
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
