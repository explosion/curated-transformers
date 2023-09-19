from typing import Any, Callable, Dict, List, Tuple, Union

from ...layers.activations import Activation
from ...util.string import StringTransform, StrLStrip, StrSub, StrSubInv
from ..hf_hub.conversion import process_hf_keys
from .config import LlamaConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    # Attention blocks.
    StrSubInv((r".self_attn", ".mha")),
    StrSubInv((r".q_proj", ".query")),
    StrSubInv((r".k_proj", ".key")),
    StrSubInv((r".v_proj", ".value")),
    StrSubInv((r".o_proj", ".output")),
    # Pointwise feedforward
    StrSubInv((r".mlp", ".ffn")),
    StrSubInv((r".up_proj", ".intermediate")),
    StrSubInv((r"ffn.down_proj", "ffn.output")),
    StrSubInv((r".gate_proj", ".gate")),
    # RMS norms
    StrSubInv((r".input_layernorm", ".attn_input_layer_norm")),
    StrSubInv((r".post_attention_layernorm", ".ffn_input_layer_norm")),
    StrSub(
        (r"^(decoder\.)?norm\.", "\\1output_layer_norm."),
        (r"^(decoder\.)?output_layer_norm\.", "\\1norm."),
    ),
    # Embeddings
    StrSubInv((r"embed_tokens.", "embeddings.piece_embeddings.")),
    StrSubInv((r"lm_head.", "output_embeddings.")),
]

DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StrLStrip("model.", reversible=False)
] + COMMON_HF_PARAM_KEY_TRANSFORMS
CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS = [
    StrSubInv((r"model.", "decoder."))
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
