from typing import Any, Callable, Dict, List, Tuple, Union

from ...layers.activations import Activation
from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import process_hf_keys
from .config import LlamaConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Attention blocks.
    StringTransformations.sub(".self_attn", ".mha"),
    StringTransformations.sub(".q_proj", ".query"),
    StringTransformations.sub(".k_proj", ".key"),
    StringTransformations.sub(".v_proj", ".value"),
    StringTransformations.sub(".o_proj", ".output"),
    # Pointwise feedforward
    StringTransformations.sub(".mlp", ".ffn"),
    StringTransformations.sub(".up_proj", ".intermediate"),
    StringTransformations.sub("ffn.down_proj", "ffn.output"),
    StringTransformations.sub(".gate_proj", ".gate"),
    # RMS norms
    StringTransformations.sub(".input_layernorm", ".attn_input_layer_norm"),
    StringTransformations.sub(".post_attention_layernorm", ".ffn_input_layer_norm"),
    StringTransformations.regex_sub(
        (r"^(decoder\.)?norm\.", "\\1output_layer_norm."),
        (r"^(decoder\.)?output_layer_norm\.", "\\1norm."),
    ),
    # Embeddings
    StringTransformations.sub("embed_tokens.", "embeddings.piece_embeddings."),
    StringTransformations.sub("lm_head.", "output_embeddings."),
]

DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.remove_prefix("model.", reversible=False)
] + COMMON_HF_PARAM_KEY_TRANSFORMS
CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.sub("model.", "decoder.")
] + COMMON_HF_PARAM_KEY_TRANSFORMS

HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "hidden_act": ("activation", Activation),
    "hidden_size": "hidden_width",
    "intermediate_size": "intermediate_width",
    "rms_norm_eps": "rms_norm_eps",
    "num_attention_heads": "n_query_heads",
    "num_hidden_layers": "n_hidden_layers",
    "vocab_size": "n_pieces",
}


def convert_hf_config(hf_config: Any) -> LlamaConfig:
    kwargs = process_hf_keys(
        "Llama", hf_config, HF_CONFIG_KEY_MAPPING, EXTRA_KWARG_KEYS
    )

    n_key_value_heads = hf_config.get("num_key_value_heads", kwargs["n_query_heads"])
    kwargs["n_key_value_heads"] = n_key_value_heads

    return LlamaConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )
