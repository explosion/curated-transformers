from typing import Any, Dict, Mapping, Optional, Set, Tuple, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Linear

from ...quantization.quantizable import Quantizable
from ..hf_hub import FromHF
from ..hf_hub.conversion import state_dict_from_hf, state_dict_to_hf
from ..transformer import TransformerCausalLM
from ._hf import CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS, _config_from_hf, _config_to_hf
from .config import FalconConfig
from .decoder import FalconDecoder

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconCausalLM")


class FalconCausalLM(
    TransformerCausalLM[FalconConfig], FromHF[FalconConfig], Quantizable
):
    """
    Falcon (`Penedo et al., 2019`_) causal language model.

    .. _Penedo et al., 2019: https://arxiv.org/abs/2306.01116
    """

    def __init__(
        self, config: FalconConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a Falcon causal LM.

        :param config:
            Causal LM configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The causal LM.
        """
        super().__init__(config)

        self.decoder = FalconDecoder(config, device=device)
        self.output_embeddings = Linear(
            in_features=config.layer.feedforward.hidden_width,
            out_features=config.embedding.n_pieces,
            bias=False,
            device=device,
        )

    @classmethod
    def state_dict_from_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_from_hf(params, CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def is_supported(cls: Type[Self], config: Dict[str, Any]) -> bool:
        return config.get("model_type") in ("falcon", "RefinedWeb", "RefinedWebModel")

    @classmethod
    def state_dict_to_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_to_hf(params, CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def config_from_hf(cls, hf_config: Mapping[str, Any]) -> FalconConfig:
        return _config_from_hf(hf_config)

    @classmethod
    def config_to_hf(cls, curated_config: FalconConfig) -> Mapping[str, Any]:
        return _config_to_hf(cls, curated_config)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = cls.config_from_hf(hf_config)
        return cls(config, device=device)

    @classmethod
    def modules_to_not_quantize(cls) -> Set[str]:
        # Ignore the output embedding matrix.
        return {"output_embeddings"}
