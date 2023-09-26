from typing import Any, Callable, Dict, List, Tuple, Union

from ...util.string import StringTransform, StringTransformations
from ..hf_hub.conversion import process_hf_keys
from .config import MPTConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

# Order-dependent.
COMMON_HF_PARAM_KEY_TRANSFORMS: List[StringTransform] = [
    StringTransformations.sub("transformer", "decoder"),
    StringTransformations.sub("blocks", "layers"),
    # Attention blocks.
    StringTransformations.sub(".attn", ".mha"),
    StringTransformations.sub(".Wqkv", ".input"),
    StringTransformations.sub(".out_proj", ".output"),
    # Pointwise feedforward.
    StringTransformations.sub(".up_proj", ".intermediate"),
    StringTransformations.sub("ffn.down_proj", "ffn.output"),
    # Layer norms.
    StringTransformations.sub(".norm_1", ".attn_input_layer_norm"),
    StringTransformations.sub(".norm_2", ".ffn_input_layer_norm"),
    StringTransformations.sub("norm_f.", "output_layer_norm."),
    # Embeddings.
    StringTransformations.sub("wte.", "embeddings.piece_embeddings."),
]


DECODER_HF_PARAM_KEY_TRANSFORMS = [
    StringTransformations.remove_prefix("transformer.", reversible=False)
] + COMMON_HF_PARAM_KEY_TRANSFORMS
CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS = COMMON_HF_PARAM_KEY_TRANSFORMS

HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "d_model": "hidden_width",
    "expansion_ratio": "intermediate_width_multiplier",
    "max_seq_len": "model_max_length",
    "n_layers": "n_hidden_layers",
    "n_heads": "n_attention_heads",
    "vocab_size": "n_pieces",
}


def convert_hf_config(hf_config: Any) -> MPTConfig:
    kwargs = process_hf_keys("MPT", hf_config, HF_CONFIG_KEY_MAPPING, EXTRA_KWARG_KEYS)

    no_bias = hf_config.get("no_bias")
    if no_bias is None:
        raise ValueError(f"Missing keys in Hugging Face model MPT config: no_bias")

    dropout_prob = 0.0
    attn_config = hf_config.get("attn_config")
    if attn_config is not None:
        dropout_prob = attn_config.get("attn_pdrop", 0.0)

    layer_norm_eps = hf_config.get("layer_norm_epsilon", 1e-5)

    return MPTConfig(
        **kwargs,
        attention_probs_dropout_prob=dropout_prob,
        hidden_dropout_prob=dropout_prob,
        layer_norm_eps=layer_norm_eps,
        use_bias=not no_bias,
    )
