import json
import re
from typing import List, Mapping

import torch
from huggingface_hub import hf_hub_download
from requests import HTTPError  # type: ignore

HF_MODEL_CONFIG = "config.json"
HF_MODEL_CHECKPOINT = "pytorch_model.bin"
HF_MODEL_SHARDED_CHECKPOINT_INDEX = "pytorch_model.bin.index.json"
HF_MODEL_SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY = "weight_map"


def get_model_config_filepath(name: str, revision: str) -> str:
    """Returns the local file path of the Hugging Face model's config.
    If the config is not found in the cache, it is downloaded from
    Hugging Face Hub.

    :param name:
        Model name.
    :param revision:
        Model revision.
    :returns:
        Absolute path to the configuration file.
    """
    try:
        return hf_hub_download(
            repo_id=name, filename=HF_MODEL_CONFIG, revision=revision
        )
    except:
        raise ValueError(
            f"Couldn't find a valid config for model `{name}` "
            f"(revision `{revision}`) on HuggingFace Model Hub"
        )


def get_model_checkpoint_filepaths(name: str, revision: str) -> List[str]:
    """Returns a list of local file paths to PyTorch checkpoints that belong
    to the Hugging Face model. In case of non-sharded models, a single file path
    is returned. In case of sharded models, multiple file paths are returned.

    If the checkpoints are not found in the cache, they are downloaded from
    Hugging Face Hub.

    :param name:
        Model name.
    :param revision:
        Model revision.
    :returns:
        List of absolute paths to the checkpoints.
    """

    # Attempt to download a non-sharded checkpoint first.
    try:
        model_filename = hf_hub_download(
            repo_id=name, filename=HF_MODEL_CHECKPOINT, revision=revision
        )
    except HTTPError:
        # We'll get a 404 HTTP error for sharded models.
        model_filename = None

    if model_filename is not None:
        return [model_filename]

    try:
        model_index_filename = hf_hub_download(
            repo_id=name, filename=HF_MODEL_SHARDED_CHECKPOINT_INDEX, revision=revision
        )
    except HTTPError:
        raise ValueError(
            f"Couldn't find a valid PyTorch checkpoint for model `{name}` "
            f"(revision `{revision}`) on HuggingFace Model Hub"
        )

    with open(model_index_filename, "r") as f:
        index = json.load(f)

    weight_map = index.get(HF_MODEL_SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY)
    if weight_map is None or not isinstance(weight_map, dict):
        raise ValueError(
            f"Invalid index file in sharded PyTorch checkpoint for model `{name}`"
        )

    filepaths = []
    # We shouldn't need to hold on to the weights map in the index as each checkpoint
    # should contain its constituent parameter names.
    for filename in set(weight_map.values()):
        resolved_filename = hf_hub_download(
            repo_id=name, filename=filename, revision=revision
        )
        filepaths.append(resolved_filename)

    return sorted(filepaths)


def _rename_old_hf_names(
    params: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    out = {}
    for name, parameter in params.items():
        name = re.sub(r"\.gamma$", ".weight", name)
        name = re.sub(r"\.beta$", ".bias", name)
        out[name] = parameter
    return out
