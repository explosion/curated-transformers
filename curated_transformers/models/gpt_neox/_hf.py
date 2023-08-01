import re
from typing import Any, Callable, Dict, Mapping, Tuple, Union

from torch import Tensor

from ...layers.activations import Activation
from ..hf_hub import _process_hf_keys
from ..module import DecoderModule
from .config import GPTNeoXConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]

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
    kwargs = _process_hf_keys(
        "GPT-NeoX", hf_config, HF_CONFIG_KEY_MAPPING, EXTRA_KWARG_KEYS
    )
    return GPTNeoXConfig(
        model_max_length=hf_config["max_position_embeddings"],
        **kwargs,
    )


def convert_hf_state_dict(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Convert state dict from HF paramater naming to ours.
    The function is insensitive to prefixes, to allow loading
    both the decoder and the full LM."""
    if issubclass(cls, DecoderModule):
        stripped_params = {
            re.sub(r"^gpt_neox\.", "", k): v
            for k, v in params.items()
            # The decoder does not the output embeddings, avoid unexpected key.
            if k != "embed_out.weight"
        }
    else:
        # Rewrap as dict if necessay to make MyPy happy.
        stripped_params = dict(params)

    out = {}
    for name, parameter in stripped_params.items():
        # These parameters are all created on-the-fly.
        if "rotary_emb" in name or "attention.bias" in name or "masked_bias" in name:
            continue

        name = name.replace("gpt_neox", "decoder")

        # Attention
        name = re.sub(r"\.attention", r".mha", name)
        name = re.sub(r"\.query_key_value", r".input", name)
        name = re.sub(r"\.mha\.dense", r".mha.output", name)

        # Pointwise feedforward
        name = re.sub(r"\.mlp", r".ffn", name)
        name = re.sub(r"\.dense_h_to_4h", r".intermediate", name)
        name = re.sub(r"\.dense_4h_to_h", r".output", name)

        # Layer norms
        name = re.sub(r"\.input_layernorm", r".attn_input_layer_norm", name)
        name = re.sub(r"\.post_attention_layernorm", r".ffn_input_layer_norm", name)
        name = re.sub(r"final_layer_norm\.", r"output_layer_norm.", name)

        # Embeddings
        name = re.sub(r"embed_in\.", r"embeddings.piece_embeddings.", name)
        name = re.sub(r"embed_out\.", r"output_embeddings.", name)

        out[name] = parameter

    return out
