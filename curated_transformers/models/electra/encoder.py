from typing import Any, Dict, Mapping, Type, TypeVar

from torch import Tensor

from ..bert import BERTConfig as ELECTRAConfig
from ..bert import BERTEncoder
from ..hf_hub.conversion import state_dict_from_hf, state_dict_to_hf
from ._hf import HF_PARAM_KEY_TRANSFORMS, _config_from_hf, _config_to_hf

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="ELECTRAEncoder")


class ELECTRAEncoder(BERTEncoder):
    """
    ELECTRA (`Clark et al., 2020`_) encoder.

    .. _Clark et al., 2020 : https://arxiv.org/abs/2003.10555
    """

    @classmethod
    def is_supported(cls: Type[Self], config: Dict[str, Any]) -> bool:
        return config.get("model_type") == "electra"

    @classmethod
    def state_dict_from_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_from_hf(params, HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def state_dict_to_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_to_hf(params, HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def config_from_hf(cls, hf_config: Mapping[str, Any]) -> ELECTRAConfig:
        return _config_from_hf(hf_config)

    @classmethod
    def config_to_hf(cls, curated_config: ELECTRAConfig) -> Mapping[str, Any]:
        return _config_to_hf(curated_config)
