from functools import partial
from typing import Any, List, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, LayerNorm, ModuleList

from ...layers.attention import AttentionHeads, AttentionMask, QkvMode, SelfAttention
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
            config.layer.hidden_width, config.layer.layer_norm_eps, device=device
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
            embedding_output=embeddings,
            layer_hidden_states=layer_outputs,
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
        return OldFalconDecoderLayer(config.layer, config.attention, device=device)

    def _create_new_decoder_architecture_layer(
        self, config: FalconConfig, device: Optional[torch.device]
    ):
        layer_norm = partial(
            LayerNorm,
            config.layer.hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )
        hidden_width = config.layer.hidden_width
        num_attention_heads = config.attention.num_query_heads
        assert config.attention.rotary_embeddings is not None
        return DecoderLayer(
            attention_layer=SelfAttention(
                attention_heads=AttentionHeads.key_value_broadcast(
                    num_query_heads=num_attention_heads,
                    num_key_value_heads=config.attention.num_key_value_heads,
                ),
                dropout_prob=config.attention.dropout_prob,
                hidden_width=hidden_width,
                qkv_mode=QkvMode.MERGED_SPLIT_AFTER,
                rotary_embeds=QueryKeyRotaryEmbeddings(
                    fraction=config.attention.rotary_embeddings.rotary_fraction,
                    base=config.attention.rotary_embeddings.rotary_base,
                    dims_per_head=hidden_width // num_attention_heads,
                ),
                use_bias=config.attention.use_bias,
                device=device,
            ),
            feed_forward_layer=PointwiseFeedForward(
                hidden_act=config.layer.hidden_act,
                hidden_width=hidden_width,
                intermediate_width=config.layer.intermediate_width,
                use_bias=config.layer.use_bias,
                use_gate=False,
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
