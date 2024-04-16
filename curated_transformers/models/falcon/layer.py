from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from ...layers.attention import (
    AttentionHeads,
    AttentionLinearBiases,
    AttentionMask,
    KeyValueCache,
    QkvMode,
    QkvSplitGroupedByKVHeads,
    ScaledDotProductAttention,
    SelfAttention,
)
from ...layers.embeddings import QueryKeyRotaryEmbeddings
from ...layers.feedforward import PointwiseFeedForward
from ..config import TransformerLayerConfig


class OldFalconDecoderLayer(Module):
    """
    Falcon (`Penedo et al., 2019`_) layer using the old decoder architecture.

    .. _Penedo et al., 2019: https://arxiv.org/abs/2306.01116
    """

    def __init__(
        self,
        layer_config: TransformerLayerConfig,
        *,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        attention_config = layer_config.attention
        if attention_config.rotary_embeddings is None:
            raise ValueError(
                "Falcon attention config does not contain rotary embedding parameters"
            )

        self.use_parallel_attention = attention_config.use_parallel_attention

        hidden_width = layer_config.feedforward.hidden_width
        n_attention_heads = attention_config.n_query_heads
        attention_biases = (
            AttentionLinearBiases(
                n_attention_heads=attention_config.n_query_heads,
                is_causal=True,
                is_inverted=True,
            )
            if attention_config.use_alibi
            else None
        )
        # Rotary embeddings are disabled when using ALiBi.
        rotary_embeds = (
            QueryKeyRotaryEmbeddings(
                fraction=attention_config.rotary_embeddings.rotary_fraction,
                base=attention_config.rotary_embeddings.rotary_base,
                head_width=hidden_width // n_attention_heads,
            )
            if not attention_config.use_alibi
            else None
        )
        self.mha = SelfAttention(
            attention_scorer=ScaledDotProductAttention(
                dropout_prob=attention_config.dropout_prob,
                linear_biases=attention_biases,
            ),
            hidden_width=hidden_width,
            attention_heads=AttentionHeads.key_value_broadcast(
                n_query_heads=attention_config.n_query_heads,
                n_key_value_heads=attention_config.n_key_value_heads,
                qkv_split=QkvSplitGroupedByKVHeads(),
            ),
            rotary_embeds=rotary_embeds,
            qkv_mode=(
                QkvMode.MERGED_SPLIT_AFTER
                if attention_config.n_key_value_heads == 1
                else QkvMode.MERGED_SPLIT_BEFORE
            ),
            use_bias=attention_config.use_bias,
            device=device,
        )
        self.attn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)
        self.attn_layer_norm = torch.nn.LayerNorm(
            hidden_width, eps=layer_config.layer_norm_eps, device=device
        )

        if not self.use_parallel_attention:
            self.ffn_layer_norm = torch.nn.LayerNorm(
                hidden_width,
                eps=layer_config.layer_norm_eps,
                device=device,
            )

        self.ffn = PointwiseFeedForward(
            activation=layer_config.feedforward.activation.module(),
            hidden_width=hidden_width,
            intermediate_width=layer_config.feedforward.intermediate_width,
            use_bias=layer_config.feedforward.use_bias,
            use_gate=layer_config.feedforward.use_gate,
            device=device,
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=layer_config.dropout_prob)

    def forward(
        self,
        input: Tensor,
        attention_mask: AttentionMask,
        *,
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
        residual = input

        attn_layer_norm = self.attn_layer_norm(input)

        attn_out, cache = self.mha(
            attn_layer_norm,
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
            use_causal_mask=True,
        )

        if self.use_parallel_attention:
            ffn_layer_norm = attn_layer_norm
        else:
            residual = residual + self.attn_output_dropout(attn_out)
            ffn_layer_norm = self.ffn_layer_norm(residual)

        ffn_out = self.ffn(ffn_layer_norm)

        if self.use_parallel_attention:
            ffn_out += attn_out

        return residual + self.ffn_output_dropout(ffn_out), cache
