from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ...layers.attention import AttentionMask
from ..bert.config import BERTAttentionConfig
from ..bert.layer import BERTAttentionConfig, BERTEncoderLayer
from .config import ALBERTLayerConfig


class ALBERTLayerGroup(Module):
    """
    ALBERT (`Lan et al., 2022`_) layer group.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    def __init__(
        self,
        layer_config: ALBERTLayerConfig,
        attention_config: BERTAttentionConfig,
        *,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        self.group_layers = ModuleList(
            [
                BERTEncoderLayer(layer_config, attention_config, device=device)
                for _ in range(layer_config.inner_group_num)
            ]
        )

    def forward(self, input: Tensor, attention_mask: AttentionMask) -> Tensor:
        """
        Apply the ALBERT layer group to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer group to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        layer_output = input
        for layer in self.group_layers:
            layer_output = layer(layer_output, attention_mask)
        return layer_output
