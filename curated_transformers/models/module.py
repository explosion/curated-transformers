from abc import abstractmethod
from typing import Generic, List, Optional
from torch import Tensor
from torch.nn import Module
from transformers.models.mctct.modeling_mctct import CausalLMOutput

from .attention import AttentionMask, CacheT
from .output import CausalLMOutputWithCache


class CausalLMModule(Generic[CacheT], Module):
    """Base class for causal language model modules."""

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

        :param input_ids: Piece identifiers to apply the language model to.
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
        :returns: Decoder representations of the given pieces and logits of
            the predicted token distribution.

        Shapes:
            input_ids, attention_mask, positions - (batch, seq_len)
        """
        raise NotImplementedError


class DecoderModule(Module):
    """Base class for decoder modules."""

    pass


class EncoderModule(Module):
    """Base class for encoder modules."""

    pass
