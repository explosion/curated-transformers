from abc import abstractmethod
from typing import Generic, List, Optional

from torch import Tensor
from torch.nn import Module

from ..layers.attention import AttentionMask
from .output import CacheT, CausalLMOutputWithCache, ModelOutput, ModelOutputWithCache


class CausalLMModule(Generic[CacheT], Module):
    """
    Base class for causal language model modules.
    """

    @abstractmethod
    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        cache: Optional[List[CacheT]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> CausalLMOutputWithCache[CacheT]:
        """
        Apply the causal language model to the given piece identifiers.

        :param piece_ids:
            Piece identifiers to apply the decoder to.

            *Shape:* ``(batch_size, seq_len)``
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
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        cache: Optional[List[CacheT]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> ModelOutputWithCache[CacheT]:
        """
        Apply the decoder to the given piece identifiers.

        :param piece_ids:
            Piece identifiers to apply the decoder to.

            *Shape:* ``(batch_size, seq_len)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param cache:
            Key/value cache to avoid recomputing key/value representations
            for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up position embeddings.
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
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        positions: Optional[Tensor] = None,
        type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        """
        Apply the encoder to the input.

        :param piece_ids:
            Piece identifiers to apply the encoder to.

            *Shape:* ``(batch_size, seq_len)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param positions:
            Input positions. Positions are used to look up position embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.

            *Shape:* ``(batch_size, seq_len)``
        :param type_ids:
            Type identifiers to indicate the spans of different
            sequences in the input. Useful when performing tasks like
            sequence classification and question answering.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Encoder output.
        """
        raise NotImplementedError
