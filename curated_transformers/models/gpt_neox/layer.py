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
from .config import GPTNeoXAttentionConfig, GPTNeoXLayerConfig


class GPTNeoXDecoderLayer(Module):
    """
    GPT-NeoX (`Black et al., 2022`_) decoder layer.

    .. _Black et al., 2022: https://arxiv.org/abs/2204.06745
    """

    def __init__(
        self,
        layer_config: GPTNeoXLayerConfig,
        attention_config: GPTNeoXAttentionConfig,
        *,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.mha = SelfAttention(
            dropout_prob=attention_config.dropout_prob,
            qkv_head_sharing=QkvHeadSharing.NONE,
            hidden_width=attention_config.hidden_width,
            num_attention_heads=attention_config.num_attention_heads,
            qkv_mode=QkvMode.MERGED_SPLIT_BEFORE,
            rotary_embeds=RotaryEmbeddingConfig(
                fraction=attention_config.rotary_fraction,
                base=attention_config.rotary_base,
            ),
            use_bias=True,
            device=device,
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.input_layer_norm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps, device=device
        )

        self.ffn = PointwiseFeedForward(
            hidden_act=layer_config.hidden_act,
            hidden_width=layer_config.hidden_width,
            intermediate_width=layer_config.intermediate_width,
            use_bias=True,
            use_gate=False,
            device=device,
        )
        self.ffn_layer_norm = torch.nn.LayerNorm(
            layer_config.hidden_width, eps=layer_config.layer_norm_eps, device=device
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
        Apply the GPT-NeoX layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        :param cache:
            Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up rotary embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Layer output and the key/value cache.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        attn_out, cache = self.mha(
            self.input_layer_norm(input),
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
            use_causal_mask=True,
        )
        attn_out = self.attn_output_dropout(attn_out)
        ffn_out = self.ffn(self.ffn_layer_norm(input))
        ffn_out = self.ffn_output_dropout(ffn_out)

        # Parallel attention & feed-forward, Section 2.1.2
        return input + attn_out + ffn_out, cache
