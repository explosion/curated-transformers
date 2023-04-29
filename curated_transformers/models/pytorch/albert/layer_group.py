from torch import Tensor
from torch.nn import Module, ModuleList

from ..attention import AttentionMask
from ..bert.config import BertAttentionConfig
from ..bert.layer import BertAttentionConfig, BertEncoderLayer
from .config import AlbertLayerConfig


class AlbertLayerGroup(Module):
    def __init__(
        self, layer_config: AlbertLayerConfig, attention_config: BertAttentionConfig
    ) -> None:
        super().__init__()

        self.group_layers = ModuleList(
            [
                BertEncoderLayer(layer_config, attention_config)
                for _ in range(layer_config.inner_group_num)
            ]
        )

    def forward(self, input: Tensor, attn_mask: AttentionMask) -> Tensor:
        """
        Shapes:
            input - (batch, seq_len, width)
        """
        layer_output = input
        for layer in self.group_layers:
            layer_output = layer(layer_output, attn_mask)
        return layer_output
