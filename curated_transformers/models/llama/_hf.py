import re
from typing import Any, Callable, Dict, Mapping, Tuple, Union

from torch import Tensor

from ...layers.activations import Activation
from ..hf_hub import _process_hf_keys
from ..module import DecoderModule
from .config import LlamaConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]


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
    kwargs = _process_hf_keys(
        "Llama", hf_config, HF_CONFIG_KEY_MAPPING, EXTRA_KWARG_KEYS
    )

    n_key_value_heads = hf_config.get("num_key_value_heads", kwargs["n_query_heads"])
    kwargs["n_key_value_heads"] = n_key_value_heads

    return LlamaConfig(
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        **kwargs,
    )


def convert_hf_state_dict(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Convert state dict from HF paramater naming to ours.
    The function is insensitive to prefixes, to allow loading
    both the decoder and the full LM."""
    if issubclass(cls, DecoderModule):
        stripped_params = {re.sub(r"^model\.", "", k): v for k, v in params.items()}
    else:
        stripped_params = {
            re.sub(r"^model\.", "decoder.", k): v for k, v in params.items()
        }

    out = {}
    for name, parameter in stripped_params.items():
        # Attention
        name = re.sub(r"\.self_attn", r".mha", name)
        name = re.sub(r"\.q_proj", r".query", name)
        name = re.sub(r"\.k_proj", r".key", name)
        name = re.sub(r"\.v_proj", r".value", name)
        name = re.sub(r"\.o_proj", r".output", name)

        # Pointwise feedforward
        name = re.sub(r"\.mlp", r".ffn", name)
        name = re.sub(r"\.up_proj", r".intermediate", name)
        name = re.sub(r"\.down_proj", r".output", name)
        name = re.sub(r"\.gate_proj", r".gate", name)

        # RMS norms
        name = re.sub(r"\.input_layernorm", r".attn_input_layer_norm", name)
        name = re.sub(r"\.post_attention_layernorm", r".ffn_input_layer_norm", name)
        name = re.sub(r"^(decoder\.)?norm\.", r"\1output_layer_norm.", name)

        # Embeddings
        name = re.sub(r"embed_tokens\.", r"embeddings.piece_embeddings.", name)
        name = re.sub(r"lm_head\.", r"output_embeddings.", name)

        out[name] = parameter

    return out
