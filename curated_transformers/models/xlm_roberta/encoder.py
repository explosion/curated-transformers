from typing import Any, Dict, Optional, Tuple

import torch

from ..roberta.config import RoBERTaConfig
from ..roberta.encoder import RoBERTaEncoder


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
