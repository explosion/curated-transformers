from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers.attention import AttentionMask
from ..layers.cache import KeyValueCache
from .module import CausalLMModule, DecoderModule, EncoderModule
from .output import CausalLMOutputWithCache, ModelOutput, ModelOutputWithCache


class TransformerDecoder(DecoderModule):
    """
    Transformer decoder (`Vaswani et al., 2017`_) base class.

    This class provides an implementation of the ``forward`` method.
    Subclasses must set the given member attributes.

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    dropout: Module
    embeddings: Module
    layers: ModuleList
    output_layer_norm: Module

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


class TransformerCausalLM(CausalLMModule[KeyValueCache]):
    """
    Transformer causal LM (`Vaswani et al., 2017`_) base class.

    This class provides an implementation of the ``forward`` method.
    Subclasses must set the given member attributes..

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    decoder: TransformerDecoder
    output_embeddings: Module

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        cache: Optional[List[KeyValueCache]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> CausalLMOutputWithCache[KeyValueCache]:
        decoder_output = self.decoder(
            input_ids,
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
        )
        if torch.jit.is_tracing():
            logits = self.output_embeddings(decoder_output[0][-1])
            return decoder_output + (logits,)  # type: ignore[return-value]
        else:
            logits = self.output_embeddings(decoder_output.last_hidden_layer_state)
            return CausalLMOutputWithCache(
                all_outputs=decoder_output.all_outputs,
                cache=decoder_output.cache,
                logits=logits,
            )


class TransformerEncoder(EncoderModule):
    """
    Transformer encoder (`Vaswani et al., 2017`_) base class.

    This class provides an implementation of the ``forward`` method.
    Subclasses must set the given member attributes.

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    embeddings: Module
    layers: ModuleList

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layer_outputs = []
        for layer in self.layers:
            layer_output, _ = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)

        return ModelOutput(all_outputs=[embeddings, *layer_outputs])
