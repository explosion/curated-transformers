import json
import warnings
from typing import Any, Dict, List, Optional

import huggingface_hub
from requests import HTTPError, ReadTimeout  # type: ignore

HF_MODEL_CONFIG = "config.json"
HF_MODEL_CHECKPOINT = "pytorch_model.bin"
HF_MODEL_SHARDED_CHECKPOINT_INDEX = "pytorch_model.bin.index.json"
HF_MODEL_SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY = "weight_map"
HF_TOKENIZER_CONFIG = "tokenizer_config.json"
SPECIAL_TOKENS_MAP = "special_tokens_map.json"
TOKENIZER_JSON = "tokenizer.json"


def get_file_metadata(
    *, filename: str, name: str, revision: str
) -> huggingface_hub.HfFileMetadata:
    """
    Get the metadata for a file on Huggingface Hub.

    :param filename:
        The file to get the metadata for.
    :param name:
        Model name.
    :param revision:
        Model revision.
    """
    url = huggingface_hub.hf_hub_url(name, filename, revision=revision)
    return huggingface_hub.get_hf_file_metadata(url)


def get_hf_config_model_type(name: str, revision: str) -> str:
    """
    Get the type of a model on Hugging Face Hub.

    :param filename:
        The file to get the type of.
    :param name:
        Model name.
    """
    config_filename = get_model_config_filepath(name, revision)
    with open(config_filename, "r") as f:
        config = json.load(f)
        model_type = config.get("model_type")
        if model_type is None:
            raise ValueError("Model type not found in Hugging Face model config")
        return model_type


def get_model_config_filepath(name: str, revision: str) -> str:
    """
    Return the local file path of the Hugging Face model's config.
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
    """
    Return a list of local file paths to PyTorch checkpoints that belong
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


def get_special_piece(
    special_tokens_map: Dict[str, Any], piece_name: str
) -> Optional[str]:
    """
    Get a special piece from the special tokens map or the tokenizer
    configuration.

    :param special_tokens_map:
        The special tokens map.
    :param piece_name:
        The piece to look up.
    :returns:
        The piece or ``None`` if this particular piece was not defined.
    """
    piece = special_tokens_map.get(piece_name)
    if isinstance(piece, dict):
        piece = piece.get("content")
    return piece


def get_special_tokens_map(*, name: str, revision="main") -> Dict[str, Any]:
    """
    Get a tokenizer's special token mapping.

    :param name:
        Model name.
    :param revision:
        Model revision.
    :returns:
        Deserialized special token_map.
    """
    return _get_and_parse_json_file(
        name=name, revision=revision, filename=SPECIAL_TOKENS_MAP
    )


def get_tokenizer_config(*, name: str, revision="main") -> Dict[str, Any]:
    """
    Get a tokenizer configuration.

    :param name:
        Model name.
    :param revision:
        Model revision.
    :returns:
        Deserialized tokenizer configuration.
    """
    return _get_and_parse_json_file(
        name=name, revision=revision, filename=HF_TOKENIZER_CONFIG
    )


def _get_and_parse_json_file(
    *, name: str, revision: str, filename: str
) -> Dict[str, Any]:
    """
    Get and parse a JSON file.

    :param name:
        Model name.
    :param revision:
        Model revision.
    :param filename:
        File to download and parse.
    :returns:
        Deserialized JSON file.
    """
    config_path = hf_hub_download(repo_id=name, filename=filename, revision=revision)
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def hf_hub_download(repo_id: str, filename: str, revision: str) -> str:
    """
    Resolve the provided filename and repository to a local file path. If the file
    is not present in the cache, it will be downloaded from the Hugging Face Hub.

    :param repo_id:
        Identifier of the source repository on Hugging Face Hub.
    :param filename:
        Name of the file in the source repository to download.
    :param revision:
        Source repository revision. Can either be a branch name
        or a SHA hash of a commit.
    :returns:
        Resolved absolute filepath.
    """

    # The HF Hub library's `hf_hub_download` function will always attempt to connect to the
    # remote repo and fetch its metadata even if it's locally cached (in order to update the
    # out-of-date artifacts in the cache). This can occasionally lead to `HTTPError/ReadTimeout`s if the
    # remote host is unreachable. Instead of failing loudly, we'll add a fallback that checks
    # the local cache for the artifacts and uses them if found.
    try:
        resolved = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision
        )
    except (HTTPError, ReadTimeout) as e:
        # Attempt to check the cache.
        resolved = huggingface_hub.try_to_load_from_cache(
            repo_id=repo_id, filename=filename, revision=revision
        )
        if resolved is None or resolved is huggingface_hub._CACHED_NO_EXIST:
            # Not found, rethrow.
            raise e
        else:
            warnings.warn(
                f"Couldn't reach Hugging Face Hub; using cached artifact for '{repo_id}@{revision}:{filename}'"
            )
    return resolved
