import re
from typing import Any, Mapping

from torch import Tensor

from ..module import DecoderModule
from .config import RefinedWebModelConfig

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]


def convert_hf_config(hf_config: Any) -> RefinedWebModelConfig:
    # Handle config options that are not set in all models.
    extra_kwargs = {k: hf_config[k] for k in EXTRA_KWARG_KEYS if k in hf_config}

    if not hf_config["parallel_attn"]:
        raise ValueError(
            "Refined Web Models without parallel attention are currently not supported"
        )

    if hf_config["alibi"]:
        raise ValueError("Refined Web Models with ALiBi are currently not supported")

    return RefinedWebModelConfig(
        hidden_width=hf_config["hidden_size"],
        layer_norm_eps=hf_config["layer_norm_epsilon"],
        multi_query=hf_config["multi_query"],
        num_hidden_layers=hf_config["n_layer"],
        num_attention_heads=hf_config["n_head"],
        rotary_embedding_base=10000,
        rotary_embedding_fraction=1.0,
        use_bias=hf_config["bias"],
        vocab_size=hf_config["vocab_size"],
        **extra_kwargs
    )


def convert_hf_state_dict(cls, params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Convert state dict from HF paramater naming to ours.
    The function is insensitive to prefixes, to allow loading
    both the decoder and the full LM."""
    if issubclass(cls, DecoderModule):
        stripped_params = {
            re.sub(r"^transformer\.", "", k): v for k, v in params.items()
        }
    else:
        stripped_params = {
            re.sub(r"^transformer\.", "decoder.", k): v for k, v in params.items()
        }

    out = {}
    for name, parameter in stripped_params.items():
        # These parameters are all created on-the-fly.
        if "rotary_emb" in name or "attention.bias" in name or "masked_bias" in name:
            continue

        name = re.sub(r"^h\.", "layers.", name)
        name = re.sub(r"decoder\.h\.", "decoder.layers.", name)

        # Attention
        name = re.sub(r"\.self_attention", r".mha", name)
        name = re.sub(r"\.query_key_value", r".input", name)
        name = re.sub(r"\.mha\.dense", r".mha.output", name)

        # Pointwise feedforward
        name = re.sub(r"\.mlp", r".ffn", name)
        name = re.sub(r"\.dense_h_to_4h", r".intermediate", name)
        name = re.sub(r"\.dense_4h_to_h", r".output", name)

        # Layer norms
        name = re.sub(r"\.input_layernorm", r".input_layer_norm", name)
        name = re.sub(r"ln_f\.", r"output_layer_norm.", name)

        # Embeddings
        name = re.sub(r"word_embeddings\.", r"embeddings.", name)
        name = re.sub(r"lm_head\.", r"output_embeddings.", name)

        out[name] = parameter

    return out
