import json
from typing import Any, Mapping, Type, TypeVar
from abc import ABC, abstractmethod
from huggingface_hub import hf_hub_download
import torch
from torch import Tensor
from torch.nn import Parameter

HF_MODEL_CONFIG = "config.json"
HF_MODEL_CHECKPOINT = "pytorch_model.bin"

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
        config_filename = hf_hub_download(
            repo_id=name, filename=HF_MODEL_CONFIG, revision=revision
        )
        with open(config_filename, "r") as f:
            config = json.load(f)
        model = cls.from_hf_config(hf_config=config)

        # Download model and convert HF parameter names to ours.
        model_filename = hf_hub_download(
            repo_id=name, filename=HF_MODEL_CHECKPOINT, revision=revision
        )
        state_dict = torch.load(model_filename, weights_only=True)
        state_dict = cls.convert_hf_state_dict(state_dict)

        model.load_state_dict(state_dict)

        return model

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Load a state dictionary.

        This method is automatically impemented by also deriving from
        `torch.nn.Module`. This mixin does not derive from `Module` in
        order to be an abstract base class.
        """
        ...
