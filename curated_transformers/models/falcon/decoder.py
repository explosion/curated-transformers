from functools import partial
from typing import Any, List, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, LayerNorm, ModuleList

from ...layers.attention import (
    AttentionHeads,
    AttentionLinearBiases,
    AttentionMask,
    QkvMode,
    SelfAttention,
)
from ...layers.cache import KeyValueCache
from ...layers.embeddings import QueryKeyRotaryEmbeddings
from ...layers.feedforward import PointwiseFeedForward
from ...layers.transformer import (
    DecoderLayer,
    TransformerDropouts,
    TransformerLayerNorms,
)
from ..hf_hub import FromHFHub
from ..module import DecoderModule
from ..output import ModelOutputWithCache
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import FalconConfig
from .layer import OldFalconDecoderLayer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconDecoder")


class FalconDecoder(DecoderModule, FromHFHub):
    """
    `Falcon`_ decoder.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    def __init__(
        self, config: FalconConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a Falcon decoder.

        :param config:
            Decoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The decoder.
        """
        super().__init__()

        self.embeddings = Embedding(
            config.embedding.vocab_size, config.embedding.embedding_width, device=device
        )
        self.dropout = Dropout(p=config.embedding.dropout_prob)

        if config.new_decoder_architecture:
            decoder_layer = partial(
                self._create_new_decoder_architecture_layer, config, device
            )
        else:
            decoder_layer = partial(
                self._create_old_decoder_architecture_layer, config, device
            )

        self.layers = ModuleList(
            [decoder_layer() for _ in range(config.layer.num_hidden_layers)]
        )

        self.output_layer_norm = LayerNorm(
            config.layer.feedforward.hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        cache: Optional[List[KeyValueCache]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> ModelOutputWithCache[KeyValueCache]:
        embeddings = self.embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        layer_output = embeddings

        layer_outputs = []
        new_cache = []
        layer_cache = None
        for layer in self.layers:
            if cache is not None:
                # The key-value cache is stored per layer, so peel off one
                # layer at a time.
                layer_cache = cache[0]
                cache = cache[1:]
            layer_output, new_layer_cache = layer(
                layer_output,
                attention_mask,
                cache=layer_cache,
                store_cache=store_cache,
                positions=positions,
            )
            layer_outputs.append(layer_output)
            if store_cache:
                new_cache.append(new_layer_cache)

        layer_outputs[-1] = self.output_layer_norm(layer_outputs[-1])

        return ModelOutputWithCache(
            all_outputs=[embeddings, *layer_outputs],
            cache=new_cache if store_cache else None,
        )

    @classmethod
    def convert_hf_state_dict(cls, params: Mapping[str, Tensor]):
        return convert_hf_state_dict(cls, params)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config, device=device)

    def _create_old_decoder_architecture_layer(
        self, config: FalconConfig, device: Optional[torch.device]
    ):
        return OldFalconDecoderLayer(config.layer, device=device)

    def _create_new_decoder_architecture_layer(
        self, config: FalconConfig, device: Optional[torch.device]
    ):
        if config.layer.attention.rotary_embeddings is None:
            raise ValueError(
                "Falcon attention config does not contain rotary embedding parameters"
            )

        hidden_width = config.layer.feedforward.hidden_width
        layer_norm = partial(
            LayerNorm,
            hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )
        num_attention_heads = config.layer.attention.num_query_heads
        attention_biases = (
            AttentionLinearBiases(
                num_attention_heads=config.layer.attention.num_query_heads,
                is_causal=True,
                is_inverted=True,
            )
            if config.layer.attention.use_alibi
            else None
        )
        # Rotary embeddings are disabled when using ALiBi.
        rotary_embeds = (
            QueryKeyRotaryEmbeddings(
                fraction=config.layer.attention.rotary_embeddings.rotary_fraction,
                base=config.layer.attention.rotary_embeddings.rotary_base,
                dims_per_head=hidden_width // num_attention_heads,
            )
            if not config.layer.attention.use_alibi
            else None
        )
        return DecoderLayer(
            attention_layer=SelfAttention(
                attention_biases=attention_biases,
                attention_heads=AttentionHeads.key_value_broadcast(
                    num_query_heads=num_attention_heads,
                    num_key_value_heads=config.layer.attention.num_key_value_heads,
                ),
                dropout_prob=config.layer.attention.dropout_prob,
                hidden_width=hidden_width,
                qkv_mode=QkvMode.MERGED_SPLIT_AFTER,
                rotary_embeds=rotary_embeds,
                use_bias=config.layer.attention.use_bias,
                device=device,
            ),
            feed_forward_layer=PointwiseFeedForward(
                activation=config.layer.feedforward.activation.module(),
                hidden_width=hidden_width,
                intermediate_width=config.layer.feedforward.intermediate_width,
                use_bias=config.layer.feedforward.use_bias,
                use_gate=config.layer.feedforward.use_gate,
                device=device,
            ),
            dropouts=TransformerDropouts.parallel_attention_dropout(
                config.layer.dropout_prob
            ),
            layer_norms=TransformerLayerNorms(
                attn_input_layer_norm=layer_norm(),
                ffn_input_layer_norm=layer_norm(),
            ),
            # The new decoder uses parallel attention unconditionally.
            parallel_attention=True,
        )
