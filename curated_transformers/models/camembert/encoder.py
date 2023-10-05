from typing import Any, Dict, Optional, Tuple

import torch

from ..roberta.config import RoBERTaConfig
from ..roberta.encoder import RoBERTaEncoder


class CamemBERTEncoder(RoBERTaEncoder):
    """
    CamemBERT (`Martin et al., 2020`_) encoder.

    .. _Martin et al., 2020: https://arxiv.org/abs/1911.03894
    """

    def __init__(self, config: RoBERTaConfig, *, device: Optional[torch.device] = None):
        """
        Construct a CamemBERT encoder.

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
        return config.get("model_type") == "camembert"
