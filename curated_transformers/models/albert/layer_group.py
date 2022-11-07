from torch import Tensor
from torch.nn import Module, ModuleList, TransformerEncoderLayer

from ..activations import _get_activation
from ..bert.config import BertAttentionConfig
from .config import AlbertLayerConfig


class AlbertLayerGroup(Module):
    def __init__(
        self, layer_config: AlbertLayerConfig, attention_config: BertAttentionConfig
    ) -> None:
        super().__init__()

        self.group_layers = ModuleList(
            [
                TransformerEncoderLayer(
                    layer_config.hidden_size,
                    attention_config.num_attention_heads,
                    layer_config.intermediate_size,
                    layer_config.dropout_prob,
                    _get_activation(layer_config.hidden_act),
                    layer_config.layer_norm_eps,
                    batch_first=True,
                )
            ]
        )

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        layer_output = input
        for layer in self.group_layers:
            layer_output = layer(layer_output, **kwargs)
        return layer_output
