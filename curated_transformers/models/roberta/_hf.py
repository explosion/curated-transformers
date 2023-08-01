import re
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Tuple, Union

from torch import Tensor

from ...layers.activations import Activation
from ..hf_hub import _process_hf_keys
from .config import RoBERTaConfig

HF_KEY_TO_CURATED_KEY = MappingProxyType(
    {
        "embeddings.word_embeddings.weight": "embeddings.piece_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.type_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.embed_output_layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.embed_output_layer_norm.bias",
    }
)


HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "pad_token_id": "padding_id",
    "attention_probs_dropout_prob": "attention_probs_dropout_prob",
    "hidden_act": ("activation", Activation),
    "hidden_dropout_prob": "hidden_dropout_prob",
    "hidden_size": "hidden_width",
    "intermediate_size": "intermediate_width",
    "layer_norm_eps": "layer_norm_eps",
    "max_position_embeddings": "max_position_embeddings",
    "num_attention_heads": "n_attention_heads",
    "num_hidden_layers": "n_hidden_layers",
    "type_vocab_size": "type_vocab_size",
    "vocab_size": "vocab_size",
}


def convert_hf_config(hf_config: Any) -> RoBERTaConfig:
    kwargs = _process_hf_keys("RoBERTa", hf_config, HF_CONFIG_KEY_MAPPING)

    return RoBERTaConfig(
        embedding_width=hf_config["hidden_size"],
        # Positions embeddings for 0..padding_id are reserved.
        model_max_length=hf_config["max_position_embeddings"]
        - (kwargs["padding_id"] + 1),
        **kwargs,
    )


def convert_hf_state_dict(params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    out = {}

    # Strip the `roberta` prefix from XLM-RoBERTa model parameters.
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
            r"\.attention\.output\.LayerNorm", r".attn_residual_layer_norm", name
        )
        name = re.sub(r"\.(intermediate)\.dense", r".ffn.\1", name)
        name = re.sub(
            r"(\.\d+)\.output\.LayerNorm", r"\1.ffn_residual_layer_norm", name
        )
        name = re.sub(r"(\.\d+)\.(output)\.dense", r"\1.ffn.\2", name)

        out[name] = parameter

    for hf_name, curated_name in HF_KEY_TO_CURATED_KEY.items():
        if hf_name in stripped_params:
            out[curated_name] = stripped_params[hf_name]

    return out
