from typing import Any, Mapping, Type, TypeVar
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Parameter

from .._compat import has_hf_transformers, transformers
from .module import CausalLMModule, DecoderModule, EncoderModule


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

        params (Mapping[str, Parameter]): the state dict to convert.
        RETURNS (Mapping[str, Tensor]): the converted state dict.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hf_config(cls: Type[Self], *, hf_config: Any) -> Self:
        """Create the module from a Hugging Face model configuration.
        state dict for the module.

        hf_config (Any): Hugging Face model configuration.
        RETURNS (Self): module constructed using the configuration.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_hub(cls: Type[Self], name: str, revision: str = "main") -> Self:
        """Construct a module and load its parameters from Hugging Face Hub.

        name (str): Model name.
        revsion (str): Model revision.
        RETURNS (Self): Module with the parameters loaded.
        """
        if not has_hf_transformers:
            raise ValueError(
                "`Loading models from Hugging Face Hub requires `transformers` package to be installed"
            )

        if issubclass(cls, CausalLMModule):
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                name, revision=revision
            )
        elif issubclass(cls, (DecoderModule, EncoderModule)):
            hf_model = transformers.AutoModel.from_pretrained(name, revision=revision)
        else:
            raise ValueError(
                "Loading from Hugging Face Hub is not supported for this module"
            )

        model = cls.from_hf_config(hf_config=hf_model.config)
        state_dict = cls.convert_hf_state_dict(hf_model.state_dict())
        model.load_state_dict(state_dict)

        return model
