from typing import Any, Dict, Mapping, Optional

import torch

from ..roberta.config import RoBERTaConfig
from ..roberta.encoder import RoBERTaEncoder
from ._hf import _config_from_hf, _config_to_hf


class XLMREncoder(RoBERTaEncoder):
    """
    XLM-RoBERTa (`Conneau et al., 2019`_) encoder.

    .. _Conneau et al., 2019: https://arxiv.org/abs/1911.02116
    """

    def __init__(self, config: RoBERTaConfig, *, device: Optional[torch.device] = None):
        """
        Construct a XLM-RoBERTa encoder.

        :param config:
            Encoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The encoder.
        """
        super().__init__(config, device=device)

    @classmethod
    def is_supported(cls, config: Dict[str, Any]) -> bool:
        return config.get("model_type") == "xlm-roberta"

    @classmethod
    def config_from_hf(cls, hf_config: Mapping[str, Any]) -> RoBERTaConfig:
        return _config_from_hf(hf_config)

    @classmethod
    def config_to_hf(cls, curated_config: RoBERTaConfig) -> Mapping[str, Any]:
        return _config_to_hf(curated_config)
