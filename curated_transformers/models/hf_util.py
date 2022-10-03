from typing import Dict
import torch
import re

from .._compat import transformers


SUPPORTED_MODEL_TYPES = ["albert", "bert", "roberta", "xlm-roberta"]


def _check_supported_hf_models(model_type: str):
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported HF model type: {model_type}")


def convert_hf_pretrained_model_parameters(
    hf_model: "transformers.PreTrainedModel",
) -> Dict[str, torch.Tensor]:
    """Converts HF model parameters to parameters that can be consumed by
    our implementation of the Transformer.

    Returns the state_dict that can be directly loaded by our Transformer module.
    """
    model_name = hf_model.config.name_or_path
    _check_supported_hf_models(hf_model.config.model_type)

    converters = {
        "albert": _convert_albert_base_state,
        "bert": _convert_bert_base_state,
        "roberta": _convert_roberta_base_state,
        "xlm-roberta": _convert_roberta_base_state,
    }

    return converters[hf_model.config.model_type](hf_model)


def _convert_albert_base_state(
    hf_model: "transformers.PreTrainedModel",
) -> Dict[str, torch.Tensor]:
    out = {}

    state_dict = dict(hf_model.state_dict().items())

    # The ALBERT encoder parameters have the following form:
    #
    # encoder.albert_layer_groups.{hidden_group}.albert_layers.{inner_layer}.{param_name}
    #
    # hidden_group is in [0, num_hidden_group)
    # inner_layer is in [0, inner_group_num)

    for name, parameter in state_dict.items():
        print(name)
        if "encoder.albert_layer" not in name:
            continue

        # TODO: Make these substitutions less ugly.

        # Remove the prefix and rename.
        name = re.sub(r"^encoder\.", "", name)

        # Layer groups
        name = re.sub(r"^albert_layer_groups\.", "groups.", name)

        # Inner layers.
        name = re.sub(r"\.albert_layers\.", ".group_layers.", name)

        # Attention blocks.
        name = re.sub(r"\.attention\.", ".mha.", name)
        name = re.sub(r"\.mha\.LayerNorm", r".attn_output_layernorm", name)
        name = re.sub(r"\.mha\.dense\.", r".mha.output.", name)

        # Pointwise feed-forward layers.
        name = re.sub(r"\.ffn\.", r".ffn.intermediate.", name)
        name = re.sub(r"\.ffn_output\.", r".ffn.output.", name)
        name = re.sub(
            r"\.full_layer_layer_norm\.",
            r".ffn_output_layernorm.",
            name,
        )

        out[name] = parameter

    # Rename and move embedding parameters to the inner BertEmbeddings module.
    out["embeddings.word_embeddings.weight"] = state_dict[
        "embeddings.word_embeddings.weight"
    ]
    out["embeddings.token_type_embeddings.weight"] = state_dict[
        "embeddings.token_type_embeddings.weight"
    ]
    out["embeddings.position_embeddings.weight"] = state_dict[
        "embeddings.position_embeddings.weight"
    ]
    out["embeddings.layer_norm.weight"] = state_dict["embeddings.LayerNorm.weight"]
    out["embeddings.layer_norm.bias"] = state_dict["embeddings.LayerNorm.bias"]

    # Embedding projection
    out["projection.weight"] = state_dict["encoder.embedding_hidden_mapping_in.weight"]
    out["projection.bias"] = state_dict["encoder.embedding_hidden_mapping_in.bias"]

    return out


def _convert_bert_base_state(
    hf_model: "transformers.PreTrainedModel",
) -> Dict[str, torch.Tensor]:
    out = {}

    state_dict = dict(hf_model.state_dict().items())

    for name, parameter in state_dict.items():
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

    # Rename and move embedding parameters to the inner BertEmbeddings module.
    out["embeddings.word_embeddings.weight"] = state_dict[
        "embeddings.word_embeddings.weight"
    ]
    out["embeddings.token_type_embeddings.weight"] = state_dict[
        "embeddings.token_type_embeddings.weight"
    ]
    out["embeddings.position_embeddings.weight"] = state_dict[
        "embeddings.position_embeddings.weight"
    ]
    out["embeddings.layer_norm.weight"] = state_dict["embeddings.LayerNorm.weight"]
    out["embeddings.layer_norm.bias"] = state_dict["embeddings.LayerNorm.bias"]

    return out


def _convert_roberta_base_state(
    hf_model: "transformers.PreTrainedModel",
) -> Dict[str, torch.Tensor]:
    out = {}

    # Strip the `roberta` prefix from XLM-Roberta model parameters.
    state_dict = {
        re.sub(r"^roberta\.", "", k): v for k, v in hf_model.state_dict().items()
    }

    for name, parameter in state_dict.items():
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

    # Rename and move embedding parameters to the inner BertEmbeddings module.
    out["embeddings.inner.word_embeddings.weight"] = state_dict[
        "embeddings.word_embeddings.weight"
    ]
    out["embeddings.inner.token_type_embeddings.weight"] = state_dict[
        "embeddings.token_type_embeddings.weight"
    ]
    out["embeddings.inner.position_embeddings.weight"] = state_dict[
        "embeddings.position_embeddings.weight"
    ]
    out["embeddings.inner.layer_norm.weight"] = state_dict[
        "embeddings.LayerNorm.weight"
    ]
    out["embeddings.inner.layer_norm.bias"] = state_dict["embeddings.LayerNorm.bias"]

    return out
