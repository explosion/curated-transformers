from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from ..attention import (
    AttentionMask,
    KeyValueCache,
    QkvHeadSharing,
    QkvMode,
    RotaryEmbeddingConfig,
    SelfAttention,
)
from ..feedforward import PointwiseFeedForward
from ..normalization import RMSNorm
from .config import LLaMAAttentionConfig, LLaMALayerConfig


class LLaMADecoderLayer(Module):
    """LLaMa (Touvron et al., 2023) layer."""

    def __init__(
        self,
        layer_config: LLaMALayerConfig,
        attention_config: LLaMAAttentionConfig,
        *,
        device: Optional[torch.device] = None
    ):
        """
        :param layer_config: Layer configuration.
        :param attention_config: Attention configuration.
        :param device: Device on which the module is to be initialized.
        """
        super().__init__()

        self.mha = SelfAttention(
            dropout_prob=attention_config.dropout_prob,
            qkv_head_sharing=QkvHeadSharing.NONE,
            hidden_width=attention_config.hidden_width,
            num_attention_heads=attention_config.num_attention_heads,
            qkv_mode=QkvMode.SEPARATE,
            rotary_embeds=RotaryEmbeddingConfig(
                fraction=attention_config.rotary_fraction,
                base=attention_config.rotary_base,
            ),
            use_bias=False,
            device=device,
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.attn_rms_norm = RMSNorm(
            layer_config.hidden_width, eps=layer_config.rms_norm_eps, device=device
        )

        self.ffn = PointwiseFeedForward(
            hidden_act=layer_config.hidden_act,
            hidden_width=layer_config.hidden_width,
            intermediate_width=layer_config.intermediate_width,
            use_bias=False,
            use_gate=True,
            device=device,
        )
        self.ffn_rms_norm = RMSNorm(
            layer_config.hidden_width, eps=layer_config.rms_norm_eps, device=device
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[AttentionMask],
        cache: Optional[KeyValueCache] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply the LLaMA layer to the given piece hidden representations.

        :param x: Hidden representations to apply the layer to.
        :param attention_mask: Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param cache: Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions: Input positions. Positions are needed to
            look up rotary embeddings. Normally, these positions are calculated
            automatically. But if the positions deviate for some reason, they
            can be provided through this argument.
        :param store_cache: Whether to cache the key/value representations for
            future reuse.
        :returns: Layer output.

        Shapes:
            x - (batch, seq_len, width)
            attention_mask - (batch, seq_len)
        """
        attn_out, cache = self.mha(
            self.attn_rms_norm(x),
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
            use_causal_mask=True,
        )
        attn_out = self.attn_output_dropout(attn_out)

        x = x + attn_out

        ffn_out = self.ffn(self.ffn_rms_norm(x))
        ffn_out = self.ffn_output_dropout(ffn_out)

        return x + ffn_out, cache
