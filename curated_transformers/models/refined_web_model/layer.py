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
from .config import RefinedWebModelAttentionConfig, RefinedWebModelLayerConfig


class RefinedWebModelDecoderLayer(Module):
    """Refined Web Model (eg. Falcon) layer."""

    def __init__(
        self,
        layer_config: RefinedWebModelLayerConfig,
        attention_config: RefinedWebModelAttentionConfig,
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
            hidden_width=attention_config.hidden_width,
            qkv_head_sharing=QkvHeadSharing.KEY_VALUE
            if attention_config.multi_query
            else QkvHeadSharing.NONE,
            num_attention_heads=attention_config.num_attention_heads,
            rotary_embeds=RotaryEmbeddingConfig(
                base=attention_config.rotary_base,
                fraction=attention_config.rotary_fraction,
            ),
            qkv_mode=QkvMode.MERGED_SPLIT_AFTER,
            use_bias=attention_config.use_bias,
            device=device,
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.input_layer_norm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps, device=device
        )

        self.ffn = PointwiseFeedForward(
            hidden_act="gelu",
            hidden_width=layer_config.hidden_width,
            intermediate_width=4 * layer_config.hidden_width,
            use_bias=layer_config.use_bias,
            device=device,
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
        Apply the GPT-NeoX layer to the given piece hidden representations.

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
        # NOTE: we currently only support parallel attention.
        x_layer_norm = self.input_layer_norm(x)

        attn_out, cache = self.mha(
            x_layer_norm,
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
            use_causal_mask=True,
        )
        attn_out = self.attn_output_dropout(attn_out)

        ffn_out = self.ffn(x_layer_norm)
        return self.ffn_output_dropout(ffn_out + attn_out) + x, cache
