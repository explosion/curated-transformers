import re
from types import MappingProxyType
from typing import Any, Mapping

from torch import Tensor

from ...util.hf import _merge_qkv, _rename_old_hf_names
from .config import BertConfig

HF_KEY_TO_CURATED_KEY = MappingProxyType(
    {
        "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
    }
)


def convert_hf_config(hf_config: Any) -> BertConfig:
    padding_id = hf_config["pad_token_id"]
    return BertConfig(
        attention_probs_dropout_prob=hf_config["attention_probs_dropout_prob"],
        embedding_width=hf_config["hidden_size"],
        hidden_act=hf_config["hidden_act"],
        hidden_dropout_prob=hf_config["hidden_dropout_prob"],
        hidden_width=hf_config["hidden_size"],
        intermediate_width=hf_config["intermediate_size"],
        layer_norm_eps=hf_config["layer_norm_eps"],
        model_max_length=hf_config["max_position_embeddings"],
        max_position_embeddings=hf_config["max_position_embeddings"],
        num_attention_heads=hf_config["num_attention_heads"],
        num_hidden_layers=hf_config["num_hidden_layers"],
        padding_id=padding_id,
        type_vocab_size=hf_config["type_vocab_size"],
        vocab_size=hf_config["vocab_size"],
    )


def convert_hf_state_dict(params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    out = {}

    renamed_params = _rename_old_hf_names(params)

    # Strip the `bert` prefix from BERT model parameters.
    stripped_params = {re.sub(r"^bert\.", "", k): v for k, v in renamed_params.items()}

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

    return _merge_qkv(out)
