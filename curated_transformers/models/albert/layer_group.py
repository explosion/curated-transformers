from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..attention import AttentionMask
from ..bert.config import BertAttentionConfig
from ..bert.layer import BertAttentionConfig, BertEncoderLayer
from .config import AlbertLayerConfig


class AlbertLayerGroup(Module):
    """
    ALBERT (Lan et al., 2022) layer group.
    """

    def __init__(
        self,
        layer_config: AlbertLayerConfig,
        attention_config: BertAttentionConfig,
        *,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        self.group_layers = ModuleList(
            [
                BertEncoderLayer(layer_config, attention_config, device=device)
                for _ in range(layer_config.inner_group_num)
            ]
        )

    def forward(self, input: Tensor, attention_mask: AttentionMask) -> Tensor:
        """
        Apply the ALBERT layer group to the input.

        :param input:
            Embeddings to apply the layer group to.
            **Shape:** (batch_size,, seq_len, width)
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
            **Shape:** (batch_size,, seq_len)
        """
        layer_output = input
        for layer in self.group_layers:
            layer_output = layer(layer_output, attention_mask)
        return layer_output
