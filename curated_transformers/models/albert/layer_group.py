from functools import partial
from typing import Optional

import torch
from torch import Tensor
from torch.nn import LayerNorm, Module, ModuleList

from ...layers.attention import (
    AttentionHeads,
    AttentionMask,
    QkvMode,
    QkvSplitGroupedByKVHeads,
    ScaledDotProductAttention,
    SelfAttention,
)
from ...layers.feedforward import PointwiseFeedForward
from ...layers.transformer import (
    EncoderLayer,
    TransformerDropouts,
    TransformerLayerNorms,
)
from .config import ALBERTLayerConfig


class ALBERTLayerGroup(Module):
    """
    ALBERT (`Lan et al., 2022`_) layer group.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    def __init__(
        self, layer_config: ALBERTLayerConfig, *, device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        layer_norm = partial(
            LayerNorm,
            layer_config.feedforward.hidden_width,
            layer_config.layer_norm_eps,
            device=device,
        )
        attention_config = layer_config.attention
        self.group_layers = ModuleList(
            [
                EncoderLayer(
                    attention_layer=SelfAttention(
                        attention_heads=AttentionHeads.uniform(
                            attention_config.n_query_heads,
                            QkvSplitGroupedByKVHeads(),
                        ),
                        attention_scorer=ScaledDotProductAttention(
                            dropout_prob=attention_config.dropout_prob,
                            linear_biases=None,
                        ),
                        hidden_width=layer_config.feedforward.hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=None,
                        use_bias=attention_config.use_bias,
                        device=device,
                    ),
                    feed_forward_layer=PointwiseFeedForward(
                        activation=layer_config.feedforward.activation.module(),
                        hidden_width=layer_config.feedforward.hidden_width,
                        intermediate_width=layer_config.feedforward.intermediate_width,
                        use_bias=layer_config.feedforward.use_bias,
                        use_gate=layer_config.feedforward.use_gate,
                        device=device,
                    ),
                    dropouts=TransformerDropouts.layer_output_dropouts(
                        layer_config.dropout_prob
                    ),
                    layer_norms=TransformerLayerNorms(
                        attn_residual_layer_norm=layer_norm(),
                        ffn_residual_layer_norm=layer_norm(),
                    ),
                    use_parallel_attention=attention_config.use_parallel_attention,
                )
                for _ in range(layer_config.n_layers_per_group)
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
        :returns:
            Layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        layer_output = input
        for layer in self.group_layers:
            layer_output = layer(layer_output, attention_mask)
        return layer_output
