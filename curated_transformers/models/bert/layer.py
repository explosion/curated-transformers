from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from ...layers.attention import AttentionMask, QkvHeadSharing, QkvMode, SelfAttention
from ...layers.feedforward import PointwiseFeedForward
from .config import BERTAttentionConfig, BERTLayerConfig


class BERTEncoderLayer(Module):
    """
    BERT (`Devlin et al., 2018`_) encoder layer.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    def __init__(
        self,
        layer_config: BERTLayerConfig,
        attention_config: BERTAttentionConfig,
        *,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.mha = SelfAttention(
            dropout_prob=attention_config.dropout_prob,
            hidden_width=attention_config.hidden_width,
            qkv_head_sharing=QkvHeadSharing.NONE,
            num_attention_heads=attention_config.num_attention_heads,
            qkv_mode=QkvMode.SEPARATE,
            rotary_embeds=None,
            use_bias=True,
            device=device,
        )
        self.attn_output_layernorm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps, device=device
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.ffn = PointwiseFeedForward(
            hidden_act=layer_config.hidden_act,
            hidden_width=layer_config.hidden_width,
            intermediate_width=layer_config.intermediate_width,
            use_bias=True,
            use_gate=False,
            device=device,
        )
        self.ffn_output_layernorm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps, device=device
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)

    def forward(self, input: Tensor, attention_mask: AttentionMask) -> Tensor:
        """
        Apply the BERT encoder layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

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
        attn_out, _ = self.mha(input, attention_mask)
        attn_out = self.attn_output_dropout(attn_out)
        attn_out = self.attn_output_layernorm(input + attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)
        ffn_out = self.ffn_output_layernorm(attn_out + ffn_out)

        return ffn_out
