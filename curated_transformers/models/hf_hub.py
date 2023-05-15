import json
from typing import Any, Iterable, Mapping, Type, TypeVar
from abc import ABC, abstractmethod
from huggingface_hub import hf_hub_download
from requests import HTTPError
import torch
from torch import Tensor
from torch.nn import Parameter

HF_MODEL_CONFIG = "config.json"
HF_MODEL_CHECKPOINT = "pytorch_model.bin"
HF_MODEL_SHARDED_CHECKPOINT_INDEX = "pytorch_model.bin.index.json"
HF_MODEL_SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY = "weight_map"

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FromPretrainedHFModel")


class FromPretrainedHFModel(ABC):
    """Mixin class for downloading models from Hugging Face Hub.

    A module using this mixin can implement the `convert_hf_state_dict`
    and `from_hf_config` methods. The mixin will then provide the
    `from_hf_hub` method to download a model from the Hugging Face Hub.
    """

    @classmethod
    @abstractmethod
    def convert_hf_state_dict(
        cls, params: Mapping[str, Parameter]
    ) -> Mapping[str, Tensor]:
        """Convert a state dict of a Hugging Face model to a valid
        state dict for the module.

        params (Mapping[str, Parameter]): The state dict to convert.
        RETURNS (Mapping[str, Tensor]): The converted state dict.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hf_config(cls: Type[Self], *, hf_config: Any) -> Self:
        """Create the module from a Hugging Face model JSON-deserialized
        model configuration.

        hf_config (Any): Hugging Face model configuration.
        RETURNS (Self): Module constructed using the configuration.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_hub(cls: Type[Self], name: str, revision: str = "main") -> Self:
        """Construct a module and load its parameters from Hugging Face Hub.

        name (str): Model name.
        revsion (str): Model revision.
        RETURNS (Self): Module with the parameters loaded.
        """
        # Download configuration and construct model.
        config_filename = _get_model_config_filepath(name, revision)
        with open(config_filename, "r") as f:
            config = json.load(f)
        model = cls.from_hf_config(hf_config=config)

        # Download model and convert HF parameter names to ours.
        checkpoint_filenames = _get_model_checkpoint_filepaths(name, revision)
        state_dict = _load_state_dict_checkpoints(checkpoint_filenames)
        state_dict = cls.convert_hf_state_dict(state_dict)

        model.load_state_dict(state_dict)

        return model

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Load a state dictionary.

        This method is automatically implemented by also deriving from
        `torch.nn.Module`. This mixin does not derive from `Module` in
        order to be an abstract base class.
        """
        ...


def _get_model_config_filepath(name: str, revision: str) -> str:
    try:
        return hf_hub_download(
            repo_id=name, filename=HF_MODEL_CONFIG, revision=revision
        )
    except:
        raise ValueError(
            f"Couldn't find a valid config for model `{name}` "
            f"(revision `{revision}`) on HuggingFace Model Hub"
        )


def _get_model_checkpoint_filepaths(name: str, revision: str) -> Iterable[str]:
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


def _load_state_dict_checkpoints(
    checkpoint_filenames: Iterable[str],
) -> Mapping[str, Any]:
    state_dict = {}
    for filename in checkpoint_filenames:
        state_dict.update(torch.load(filename, weights_only=True))
    return state_dict
