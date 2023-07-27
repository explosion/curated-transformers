from abc import abstractmethod
from typing import Generic, List, Optional

from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers.attention import AttentionMask
from ..layers.cache import KeyValueCache
from .output import CacheT, CausalLMOutputWithCache, ModelOutput, ModelOutputWithCache


class CausalLMModule(Generic[CacheT], Module):
    """
    Base class for causal language model modules.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        cache: Optional[List[CacheT]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> CausalLMOutputWithCache[CacheT]:
        """
        Apply the causal language model to the given piece identifiers.

        :param input_ids:
            Piece identifiers to apply the decoder to.

            *Shape:* ``(batch_size, seq_len)``
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
            positions deviate for some reason, they can be provided through this argument.

            *Shape:* ``(batch_size, seq_len)``
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Causal language model output with key/value cache.
        """
        raise NotImplementedError


class DecoderModule(Generic[CacheT], Module):
    """
    Base class for decoder modules.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        cache: Optional[List[CacheT]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> ModelOutputWithCache[CacheT]:
        """
        Apply the decoder to the given piece identifiers.

        :param input_ids:
            Piece identifiers to apply the decoder to.

            *Shape:* ``(batch_size, seq_len)``
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
            positions deviate for some reason, they can be provided through this argument.

            *Shape:* ``(batch_size, seq_len)``
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Decoder output with key/value cache.
        """
        raise NotImplementedError


class EncoderModule(Module):
    """
    Base class for encoder modules.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        """
        Apply the encoder to the input.

        :param input_ids:
            Piece identifiers to apply the encoder to.

            *Shape:* ``(batch_size, seq_len)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        :param token_type_ids:
            Token type identifiers to indicate the spans of different
            sequences in the input. Useful when performing tasks like
            sequence classification and question answering.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Encoder output.
        """
        raise NotImplementedError


class TransformerDecoder(DecoderModule):
    """
    Transformer decoder (`Vaswani et al., 2017`_) base class.

    This class provides an implementation of the forward method. Deriving
    classes must set the given member variables.

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


class TransformerEncoder(EncoderModule):
    """
    Transformer encoder (`Vaswani et al., 2017`_) base class.

    This class provides an implementation of the forward method. Deriving
    classes must set the given member variables.

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
        token_type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layer_outputs = []
        for layer in self.layers:
            layer_output, _ = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)

        return ModelOutput(all_outputs=[embeddings, *layer_outputs])
