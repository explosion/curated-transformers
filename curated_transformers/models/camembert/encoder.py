from typing import Optional

import torch

from ..roberta.config import RoBERTaConfig
from ..roberta.encoder import RoBERTaEncoder


class CamemBERTEncoder(RoBERTaEncoder):
    """
    CamemBERT (Martin et al., 2020) encoder.
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
