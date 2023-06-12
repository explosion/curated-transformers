import json
from typing import Any, Iterable, List, Mapping, Optional, Type, TypeVar
from abc import ABC, abstractmethod
from huggingface_hub import hf_hub_download
from requests import HTTPError  # type: ignore
import torch
from torch import Tensor

from .util.serde import load_model_from_checkpoints, DeserializationParamBucket

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
        cls, params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        """Convert a state dict of a Hugging Face model to a valid
        state dict for the module.

        :param params:
            The state dict to convert.
        :returns:
            The converted state dict.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        """Create the module from a Hugging Face model JSON-deserialized
        model configuration.

        :param hf_config:
            Hugging Face model configuration.
        :param device:
            Device on which to initialize the model.
        :returns:
            Module constructed using the configuration.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_hub(
        cls: Type[Self],
        name: str,
        revision: str = "main",
        *,
        device: Optional[torch.device] = None,
    ) -> Self:
        """Construct a module and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revsion:
            Model revision.
        :param device:
            Device on which to initialize the model.
        :returns:
            Module with the parameters loaded.
        """
        # Download configuration and construct model.
        config_filename = _get_model_config_filepath(name, revision)
        with open(config_filename, "r") as f:
            config = json.load(f)
        # Initialize the model on the torch `meta` device to avoid unnecessary allocations.
        model = cls.from_hf_config(hf_config=config, device=torch.device("meta"))

        # Convert the model to the expected dtype.
        dtype_str = config.get("torch_dtype")
        if dtype_str is not None:
            dtype = getattr(torch, dtype_str, None)
            if dtype is None or not isinstance(dtype, torch.dtype):
                raise ValueError(f"Invalid torch dtype `{dtype_str}`")
            model.to(dtype=dtype)

        # Download model and convert HF parameter names to ours.
        checkpoint_filenames = _get_model_checkpoint_filepaths(name, revision)
        load_model_from_checkpoints(
            model,  # type:ignore
            filepaths=checkpoint_filenames,
            deserialization_buckets=model.deserialization_param_buckets(),
            state_dict_converter=cls.convert_hf_state_dict,
            device=device,
        )

        # Ensure that any non-persistent buffers are also moved to
        # the correct device.
        if device is not None:
            model.to(device)

        return model

    @abstractmethod
    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ):
        """Moves and/or casts the parameters and buffers.

        This method is automatically implemented by also deriving from
        `torch.nn.Module`. This mixin does not derive from `Module` in
        order to be an abstract base class.
        """
        ...

    @abstractmethod
    def deserialization_param_buckets(self) -> List[DeserializationParamBucket]:
        """Returns a list of buckets into which parameters are sorted
        during loading. Each bucket represents a group of parameters
        that need to be deserialized together.
        """
        raise NotImplementedError


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
