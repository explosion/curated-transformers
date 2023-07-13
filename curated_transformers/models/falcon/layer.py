from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from ...layers.attention import (
    AttentionMask,
    KeyValueCache,
    QkvHeadSharing,
    QkvMode,
    RotaryEmbeddingConfig,
    SelfAttention,
)
from ...layers.feedforward import PointwiseFeedForward
from .config import FalconAttentionConfig, FalconLayerConfig


class FalconDecoderLayer(Module):
    """
    `Falcon`_ layer.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    def __init__(
        self,
        layer_config: FalconLayerConfig,
        attention_config: FalconAttentionConfig,
        *,
        device: Optional[torch.device] = None
    ):
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
            use_gate=False,
            device=device,
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[AttentionMask],
        cache: Optional[KeyValueCache] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply the Falcon layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        :param cache:
            Key/value cache to avoid recomputing key/value representations
            for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up rotary embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this
            argument.
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Layer output and the key/value cache.
        """
        # NOTE: we currently only support parallel attention.
        input_layer_norm = self.input_layer_norm(input)

        attn_out, cache = self.mha(
            input_layer_norm,
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
            use_causal_mask=True,
        )
        attn_out = self.attn_output_dropout(attn_out)

        ffn_out = self.ffn(input_layer_norm)
        return self.ffn_output_dropout(ffn_out + attn_out) + input, cache
