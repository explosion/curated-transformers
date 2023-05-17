from typing import Any, Mapping
import re
from torch import Tensor
from torch.nn import Parameter

from .config import GPTNeoXConfig
from ..module import DecoderModule

ATTENTION_DROPOUT = "attention_probs_dropout_prob"
HIDDEN_DROPOUT = "hidden_dropout_prob"
EXTRA_KWARG_KEYS = [ATTENTION_DROPOUT, HIDDEN_DROPOUT]


def convert_hf_config(hf_config: Any) -> GPTNeoXConfig:
    # Handle config options that are not set in all models.
    extra_kwargs = {
        k: hf_config[EXTRA_KWARG_KEYS] for k in EXTRA_KWARG_KEYS if k in hf_config
    }

    return GPTNeoXConfig(
        hidden_act=hf_config["hidden_act"],
        hidden_width=hf_config["hidden_size"],
        intermediate_width=hf_config["intermediate_size"],
        layer_norm_eps=hf_config["layer_norm_eps"],
        max_position_embeddings=hf_config["max_position_embeddings"],
        model_max_length=hf_config["max_position_embeddings"],
        num_attention_heads=hf_config["num_attention_heads"],
        num_hidden_layers=hf_config["num_hidden_layers"],
        rotary_embedding_base=hf_config["rotary_emb_base"],
        rotary_embedding_fraction=hf_config["rotary_pct"],
        vocab_size=hf_config["vocab_size"],
        **extra_kwargs
    )


def convert_hf_state_dict(cls, params: Mapping[str, Parameter]) -> Mapping[str, Tensor]:
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
        name = re.sub(r"\.input_layernorm", r".mha_layer_norm", name)
        name = re.sub(r"\.post_attention_layernorm", r".ffn_layer_norm", name)
        name = re.sub(r"final_layer_norm\.", r"output_layer_norm.", name)

        # Embeddings
        name = re.sub(r"embed_in\.", r"embeddings.", name)
        name = re.sub(r"embed_out\.", r"output_embeddings.", name)

        out[name] = parameter

    return out
