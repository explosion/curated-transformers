from typing import Dict, OrderedDict
import torch
import re


from .albert.encoder import AlbertEncoder
from .bert.encoder import BertEncoder
from .curated_transformer import CuratedTransformer, CuratedEncoderT
from .roberta.encoder import RobertaEncoder
from .._compat import transformers
from ..errors import Errors

SUPPORTED_MODEL_TYPES = ["albert", "bert", "camembert", "roberta", "xlm-roberta"]
SUPPORTED_CURATED_ENCODERS = (AlbertEncoder, BertEncoder, RobertaEncoder)


def _check_supported_hf_models(model_type: str):
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            Errors.E007.format(
                unsupported_model=model_type, supported_models=SUPPORTED_MODEL_TYPES
            )
        )


def convert_pretrained_model_for_encoder(
    transformer: CuratedTransformer[CuratedEncoderT],
    params: OrderedDict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Converts parameters from a compatible pre-trained model to
    parameters that can be consumed by the given curated transformer.

    Returns the state_dict that can be directly loaded by the curated
    transformer.
    """
    params = _rename_old_hf_names(params)
    encoder = transformer.curated_encoder

    if isinstance(encoder, AlbertEncoder):
        converted = _convert_albert_base_state(params)
    elif isinstance(encoder, BertEncoder):
        converted = _convert_bert_base_state(params)
    elif isinstance(encoder, RobertaEncoder):
        converted = _convert_roberta_base_state(params)
    else:
        raise TypeError(
            Errors.E026.format(
                unsupported_encoder=type(encoder),
                supported_encoders=SUPPORTED_CURATED_ENCODERS,
            )
        )

    return _add_curated_encoder_prefix(converted)


def convert_hf_pretrained_model_parameters(
    hf_model: "transformers.PreTrainedModel",
) -> Dict[str, torch.Tensor]:
    """Converts HF model parameters to parameters that can be consumed by
    our implementation of the Transformer.

    Returns the state_dict that can be directly loaded by one of the
    curated transformers.
    """
    _check_supported_hf_models(hf_model.config.model_type)  # type: ignore

    converters = {
        "albert": _convert_albert_base_state,
        "bert": _convert_bert_base_state,
        "camembert": _convert_roberta_base_state,
        "roberta": _convert_roberta_base_state,
        "xlm-roberta": _convert_roberta_base_state,
    }

    converted = converters[hf_model.config.model_type](hf_model.state_dict())  # type: ignore

    return _add_curated_encoder_prefix(converted)


def _add_curated_encoder_prefix(
    converted: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {f"curated_encoder.{k}": v for k, v in converted.items()}


def _rename_old_hf_names(
    params: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    out = OrderedDict()
    for name, parameter in params.items():
        name = re.sub(r"\.gamma$", ".weight", name)
        name = re.sub(r"\.beta$", ".bias", name)
        out[name] = parameter
    return out


def _convert_albert_base_state(
    params: OrderedDict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    # Strip the `albert` prefix from ALBERT model parameters.
    stripped_params = {re.sub(r"^albert\.", "", k): v for k, v in params.items()}

    # The ALBERT encoder parameters have the following form:
    #
    # encoder.albert_layer_groups.{hidden_group}.albert_layers.{inner_layer}.{param_name}
    #
    # hidden_group is in [0, num_hidden_group)
    # inner_layer is in [0, inner_group_num)

    out = {}
    for name, parameter in stripped_params.items():
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
    out["embeddings.word_embeddings.weight"] = stripped_params[
        "embeddings.word_embeddings.weight"
    ]
    out["embeddings.token_type_embeddings.weight"] = stripped_params[
        "embeddings.token_type_embeddings.weight"
    ]
    out["embeddings.position_embeddings.weight"] = stripped_params[
        "embeddings.position_embeddings.weight"
    ]
    out["embeddings.layer_norm.weight"] = stripped_params["embeddings.LayerNorm.weight"]
    out["embeddings.layer_norm.bias"] = stripped_params["embeddings.LayerNorm.bias"]

    # Embedding projection
    out["embeddings.projection.weight"] = stripped_params[
        "encoder.embedding_hidden_mapping_in.weight"
    ]
    out["embeddings.projection.bias"] = stripped_params[
        "encoder.embedding_hidden_mapping_in.bias"
    ]

    return _merge_qkv_albert(out)


def _convert_bert_base_state(
    params: OrderedDict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = {}

    # Strip the `bert` prefix from BERT model parameters.
    stripped_params = {re.sub(r"^bert\.", "", k): v for k, v in params.items()}

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

    # Rename and move embedding parameters to the inner BertEmbeddings module.
    out["embeddings.word_embeddings.weight"] = stripped_params[
        "embeddings.word_embeddings.weight"
    ]
    out["embeddings.token_type_embeddings.weight"] = stripped_params[
        "embeddings.token_type_embeddings.weight"
    ]
    out["embeddings.position_embeddings.weight"] = stripped_params[
        "embeddings.position_embeddings.weight"
    ]
    out["embeddings.layer_norm.weight"] = stripped_params["embeddings.LayerNorm.weight"]
    out["embeddings.layer_norm.bias"] = stripped_params["embeddings.LayerNorm.bias"]

    return _merge_qkv(out)


def _convert_roberta_base_state(
    params: OrderedDict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
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

    # Rename and move embedding parameters to the inner BertEmbeddings module.
    out["embeddings.inner.word_embeddings.weight"] = stripped_params[
        "embeddings.word_embeddings.weight"
    ]
    out["embeddings.inner.token_type_embeddings.weight"] = stripped_params[
        "embeddings.token_type_embeddings.weight"
    ]
    out["embeddings.inner.position_embeddings.weight"] = stripped_params[
        "embeddings.position_embeddings.weight"
    ]
    out["embeddings.inner.layer_norm.weight"] = stripped_params[
        "embeddings.LayerNorm.weight"
    ]
    out["embeddings.inner.layer_norm.bias"] = stripped_params[
        "embeddings.LayerNorm.bias"
    ]

    return _merge_qkv(out)


def _merge_qkv(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for name, parameter in params.items():
        m = re.match(
            r"layers\.(?P<layer>[0-9]+)\.mha\.(query|key|value).(?P<param_type>weight|bias)",
            name,
        )
        if m:
            if "query" in name:
                base = f"layers.{m['layer']}.mha"
                out[f"{base}.input.{m['param_type']}"] = torch.cat(
                    [
                        parameter,
                        params[f"{base}.key.{m['param_type']}"],
                        params[f"{base}.value.{m['param_type']}"],
                    ]
                )
            continue
        out[name] = parameter

    return out


def _merge_qkv_albert(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for name, parameter in params.items():
        m = re.match(
            r"groups\.(?P<group>[0-9]+)\.group_layers\.(?P<layer>[0-9]+)\.mha\.(query|key|value).(?P<param_type>weight|bias)",
            name,
        )
        if m:
            if "query" in name:
                base = f"groups.{m['group']}.group_layers.{m['layer']}.mha"
                out[f"{base}.input.{m['param_type']}"] = torch.cat(
                    [
                        parameter,
                        params[f"{base}.key.{m['param_type']}"],
                        params[f"{base}.value.{m['param_type']}"],
                    ]
                )
            continue
        out[name] = parameter

    return out
