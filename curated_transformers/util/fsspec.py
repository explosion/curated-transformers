import json
import os
from typing import Any, Dict, List, Optional, Tuple

from fsspec import AbstractFileSystem

from .._compat import has_safetensors
from .hf import (
    HF_MODEL_CONFIG,
    HF_TOKENIZER_CONFIG,
    PRIMARY_CHECKPOINT_FILENAMES,
    SHARDED_CHECKPOINT_INDEX_FILENAMES,
    SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY,
    SPECIAL_TOKENS_MAP,
)
from .serde import (
    _MODEL_CHECKPOINT_TYPE,
    FsspecModelFile,
    ModelCheckpointType,
    ModelFile,
)


def get_file_metadata(
    *,
    fs: AbstractFileSystem,
    model_path: str,
    filename: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a file from a model on an fsspec filesystem.

    :param fs:
        The filesystem on which the model is stored.
    :param model_path:
        The path of the model on the filesystem.
    :param filename:
        The file to get metadata for.
    :param fsspec_args:
        Implementation-specific keyword arguments to pass to fsspec
        filesystem operations.
    :returns:
        File metadata as a dictionary or ``None`` if the file does not
        exist.

    """
    index = get_path_index(fs, model_path, fsspec_args=fsspec_args)
    return index.get(filename)


def get_model_config(
    fs: AbstractFileSystem,
    model_path: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get the configuation of a model on an fsspec filesystem.

    :param fs:
        The filesystem on which the model is stored.
    :param model_path:
        The path of the model on the filesystem.
    :param fsspec_args:
        Implementation-specific keyword arguments to pass to fsspec
        filesystem operations.
    :returns:
        The model configuration.
    """
    config = _get_and_parse_json_file(
        fs,
        path=f"{model_path}/{HF_MODEL_CONFIG}",
        fsspec_args=fsspec_args,
    )
    if config is None:
        raise ValueError(
            f"Cannot open model config path: {model_path}/{HF_MODEL_CONFIG}"
        )
    return config


def get_config_model_type(
    fs: AbstractFileSystem,
    model_path: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Get the type of a model on an fsspec filesystem.

    :param fs:
        The filesystem on which the model is stored.
    :param model_path:
        The path of the model on the filesystem.
    :param fsspec_args:
        Implementation-specific keyword arguments to pass to fsspec
        filesystem operations.
    :returns:
        The model type.
    """
    config = get_model_config(fs, model_path, fsspec_args=fsspec_args)
    return config.get("model_type")


def get_tokenizer_config(
    fs: AbstractFileSystem,
    model_path: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the configuration of a tokenizer on an fsspec filesystem.

    :param fs:
        The filesystem on which the model is stored.
    :param model_path:
        The path of the model on the filesystem.
    :param fsspec_args:
        Implementation-specific keyword arguments to pass to fsspec
        filesystem operations.
    :returns:
        Deserialized tokenizer configuration.
    """
    return _get_and_parse_json_file(
        fs,
        path=f"{model_path}/{HF_TOKENIZER_CONFIG}",
        fsspec_args=fsspec_args,
    )


def get_special_tokens_map(
    fs: AbstractFileSystem,
    model_path: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the special token mapping of a tokenizer on an fsspec filesystem.

    :param fs:
        The filesystem on which the model is stored.
    :param model_path:
        The path of the model on the filesystem.
    :param fsspec_args:
        Implementation-specific keyword arguments to pass to fsspec
        filesystem operations.
    :returns:
        Deserialized special token_map.
    """
    return _get_and_parse_json_file(
        fs, path=f"{model_path}/{SPECIAL_TOKENS_MAP}", fsspec_args=fsspec_args
    )


def _get_and_parse_json_file(
    fs: AbstractFileSystem,
    *,
    path: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a JSON file from an fsspec filesystem and parse it.

    :param fs:
        The filesystem on which the model is stored.
    :param path:
        The path of the JSON file.
    :param fsspec_args:
        Implementation-specific keyword arguments to pass to fsspec
        filesystem operations.
    :returns:
        List of absolute paths to the checkpoints
        and the checkpoint type.
    """
    fsspec_args = {} if fsspec_args is None else fsspec_args

    if not fs.exists(path, **fsspec_args):
        return None

    with fs.open(path, "r", encoding="utf-8", **fsspec_args) as f:
        config = json.load(f)
    return config


def get_model_checkpoint_files(
    fs: AbstractFileSystem,
    model_path: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ModelFile], ModelCheckpointType]:
    """
    Return a list of local file paths to checkpoints that belong to the model
    on an fsspec filesystem. In case of non-sharded models, a single file path
    is returned. In case of sharded models, multiple file paths are returned.

    :param fs:
        The filesystem on which the model is stored.
    :param model_path:
        The path of the model on the filesystem.
    :param fsspec_args:
        Implementation-specific keyword arguments to pass to fsspec
        filesystem operations.
    :returns:
        List of absolute paths to the checkpoints
        and the checkpoint type.
    """
    fsspec_args = {} if fsspec_args is None else fsspec_args

    def get_checkpoint_paths(
        checkpoint_type: ModelCheckpointType,
    ) -> List[ModelFile]:
        index = get_path_index(fs, model_path, fsspec_args=fsspec_args)

        # Attempt to download a non-sharded checkpoint first.
        entry = index.get(PRIMARY_CHECKPOINT_FILENAMES[checkpoint_type])
        if entry is not None:
            return [FsspecModelFile(fs, entry["name"], fsspec_args)]

        # Try sharded checkpoint.
        index_filename = SHARDED_CHECKPOINT_INDEX_FILENAMES[checkpoint_type]
        entry = index.get(index_filename)
        if entry is None:
            raise ValueError(
                f"Couldn't find a valid {checkpoint_type.pretty_name} checkpoint for "
                f"model with path `{model_path}`. Could not open {index_filename}"
            )

        with fs.open(entry["name"], "rb", **fsspec_args) as f:
            index = json.load(f)

        weight_map = index.get(SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY)
        if not isinstance(weight_map, dict):
            raise ValueError(
                f"Invalid index file in sharded {checkpoint_type.pretty_name} "
                f"checkpoint for model with path `{model_path}`"
            )

        filepaths = []
        # We shouldn't need to hold on to the weights map in the index as each checkpoint
        # should contain its constituent parameter names.
        for filename in set(weight_map.values()):
            filepaths.append(f"{model_path}/{filename}")

        return [FsspecModelFile(fs, path, fsspec_args) for path in sorted(filepaths)]

    checkpoint_type = _MODEL_CHECKPOINT_TYPE.get()
    checkpoint_paths: Optional[List[ModelFile]] = None

    if checkpoint_type is None:
        # Precedence: Safetensors > PyTorch
        if has_safetensors:
            try:
                checkpoint_type = ModelCheckpointType.SAFE_TENSORS
                checkpoint_paths = get_checkpoint_paths(checkpoint_type)
            except ValueError:
                pass
        if checkpoint_paths is None:
            checkpoint_type = ModelCheckpointType.PYTORCH_STATE_DICT
            checkpoint_paths = get_checkpoint_paths(checkpoint_type)
    else:
        checkpoint_paths = get_checkpoint_paths(checkpoint_type)

    assert checkpoint_paths is not None
    assert checkpoint_type is not None
    return checkpoint_paths, checkpoint_type


def get_path_index(
    fs: AbstractFileSystem,
    path: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Get the files and their metadata of a model on an fsspec filesystem.

    :param fs:
        The filesystem on which the model is stored.
    :param path:
        The path to return the index for.
    :param fsspec_args:
        Implementation-specific keyword arguments to pass to fsspec
        filesystem operations.
    :returns:
        List of absolute paths to the checkpoints
        and the checkpoint type.
    """
    fsspec_args = {} if fsspec_args is None else fsspec_args

    try:
        return {
            os.path.basename(entry["name"]): entry
            for entry in fs.ls(path, **fsspec_args)
        }
    except FileNotFoundError:
        raise ValueError(f"Path cannot be found: {path}")
