import re
from typing import Any, Callable, Dict, Mapping, Tuple, Union

from torch import Tensor, dropout, layer_norm

from ..hf_hub import _process_hf_keys
from ..module import DecoderModule
from .config import MPTConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "d_model": "hidden_width",
    "expansion_ratio": "intermediate_width_multiplier",
    "max_seq_len": "model_max_length",
    "n_layers": "n_hidden_layers",
    "n_heads": "n_attention_heads",
    "vocab_size": "n_pieces",
}


def convert_hf_config(hf_config: Any) -> MPTConfig:
    kwargs = _process_hf_keys("MPT", hf_config, HF_CONFIG_KEY_MAPPING, EXTRA_KWARG_KEYS)

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


def convert_hf_state_dict(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Convert state dict from HF paramater naming to ours.
    The function is insensitive to prefixes, to allow loading
    both the decoder and the full LM."""
    if issubclass(cls, DecoderModule):
        stripped_params = {
            re.sub(r"^transformer\.", "", k): v
            for k, v in params.items()
            # The decoder does not the output embeddings, avoid unexpected key.
            if k != "lm_head.weight"
        }
    else:
        # Rewrap as dict if necessay to make MyPy happy.
        stripped_params = dict(params)

    out = {}
    for name, parameter in stripped_params.items():
        # Input and output embeddings are tied in MPT.
        if "lm_head" in name:
            continue

        name = name.replace("transformer", "decoder")
        name = name.replace("blocks", "layers")

        # Attention
        name = re.sub(r"\.attn", r".mha", name)
        name = re.sub(r"\.Wqkv", r".input", name)
        name = re.sub(r"\.out_proj", r".output", name)

        # Pointwise feedforward
        name = re.sub(r"\.up_proj", r".intermediate", name)
        name = re.sub(r"\.down_proj", r".output", name)

        # Layer norms
        name = re.sub(r"\.norm_1", r".attn_input_layer_norm", name)
        name = re.sub(r"\.norm_2", r".ffn_input_layer_norm", name)
        name = re.sub(r"norm_f\.", r"output_layer_norm.", name)

        # Embeddings
        name = re.sub(r"wte\.", r"embeddings.piece_embeddings.", name)

        out[name] = parameter

    return out
