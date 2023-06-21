import re
from types import MappingProxyType
from typing import Any, Mapping

from torch import Tensor

from .config import RobertaConfig

HF_KEY_TO_CURATED_KEY = MappingProxyType(
    {
        "embeddings.word_embeddings.weight": "embeddings.inner.word_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.inner.token_type_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.inner.position_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.inner.layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.inner.layer_norm.bias",
    }
)


EXPECTED_CONFIG_KEYS = {
    "pad_token_id",
    "attention_probs_dropout_prob",
    "hidden_act",
    "hidden_dropout_prob",
    "hidden_size",
    "intermediate_size",
    "layer_norm_eps",
    "max_position_embeddings",
    "num_attention_heads",
    "num_hidden_layers",
    "type_vocab_size",
    "vocab_size",
}


def convert_hf_config(hf_config: Any) -> RobertaConfig:
    missing_keys = tuple(sorted(EXPECTED_CONFIG_KEYS.difference(set(hf_config.keys()))))
    if len(missing_keys) != 0:
        raise ValueError(f"Missing keys in HF RoBERTa model config: {missing_keys}")

    padding_id = hf_config["pad_token_id"]
    return RobertaConfig(
        attention_probs_dropout_prob=hf_config["attention_probs_dropout_prob"],
        embedding_width=hf_config["hidden_size"],
        hidden_act=hf_config["hidden_act"],
        hidden_dropout_prob=hf_config["hidden_dropout_prob"],
        hidden_width=hf_config["hidden_size"],
        intermediate_width=hf_config["intermediate_size"],
        layer_norm_eps=hf_config["layer_norm_eps"],
        # Positions embeddings for 0..padding_id are reserved.
        model_max_length=hf_config["max_position_embeddings"] - (padding_id + 1),
        max_position_embeddings=hf_config["max_position_embeddings"],
        num_attention_heads=hf_config["num_attention_heads"],
        num_hidden_layers=hf_config["num_hidden_layers"],
        padding_id=padding_id,
        type_vocab_size=hf_config["type_vocab_size"],
        vocab_size=hf_config["vocab_size"],
    )


def convert_hf_state_dict(params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    out = {}

    # Strip the `roberta` prefix from XLM-Roberta model parameters.
    stripped_params = {re.sub(r"^roberta\.", "", k): v for k, v in params.items()}

    for name, parameter in stripped_params.items():
        if "encoder.layer." not in name:
            continue

        # TODO: Make these substitutions less ugly.

        # Remove the prefix and rename the internal 'layers' variable.
        name = re.sub(r"^encoder\.", "", name)
        name = re.sub(r"^layer", "layers", name)

        # The HF model has one more level of indirection for the output layers in their
        # attention heads and the feed-forward network layers.
        name = re.sub(r"\.attention\.self\.(query|key|value)", r".mha.\1", name)
        name = re.sub(r"\.attention\.(output)\.dense", r".mha.\1", name)
        name = re.sub(
            r"\.attention\.output\.LayerNorm", r".attn_output_layernorm", name
        )
        name = re.sub(r"\.(intermediate)\.dense", r".ffn.\1", name)
        name = re.sub(r"(\.\d+)\.output\.LayerNorm", r"\1.ffn_output_layernorm", name)
        name = re.sub(r"(\.\d+)\.(output)\.dense", r"\1.ffn.\2", name)

        out[name] = parameter

    for hf_name, curated_name in HF_KEY_TO_CURATED_KEY.items():
        if hf_name in stripped_params:
            out[curated_name] = stripped_params[hf_name]

    return out
