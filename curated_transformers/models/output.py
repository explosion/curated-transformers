from dataclasses import dataclass
from typing import Generic, List, Optional, Protocol, TypeVar

import torch
from torch import Tensor

CacheProtocolSelf = TypeVar("CacheProtocolSelf", bound="CacheProtocol")


class CacheProtocol(Protocol):
    def filter_batch_items(self: CacheProtocolSelf, mask: Tensor) -> CacheProtocolSelf:
        """
        Filter batch sequences from the cache.

        Sequences for which the mask is ``True`` are retained.

        :param mask:
            Mask of batch items to retain.
            **Shape:** (batch_size,,)
        :returns:
            Filtered items.
        """
        ...


CacheT = TypeVar("CacheT", bound=CacheProtocol)


@dataclass
class KeyValueCache:
    """
    Cache type for layers that cache keys and values.
    """

    key: Tensor
    value: Tensor

    def filter_batch_items(self, mask: Tensor) -> "KeyValueCache":
        if mask.ndim != 1:
            raise ValueError(
                f"Cache mask must be a 1D tensor, has {mask.ndim} dimensions."
            )
        if mask.size(0) != self.key.size(0):
            raise ValueError(
                f"Cache mask size ({mask.size(0)}) must match cache batch size ({self.key.size(0)})."
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"Cache mask dtype must be bool, was: {mask.dtype}.")

        return KeyValueCache(key=self.key[mask], value=self.value[mask])


@torch.jit.script
class ModelOutput:
    """
    Base class for model outputs.
    """

    # The first element is the output of the embedding layer with shape (batch, seq, width).
    # The rest of the elements are the states of each encoder hidden layer respectively with shape (batch, seq, width).
    all_outputs: List[Tensor]

    def __init__(
        self, *, embedding_output: Tensor, layer_hidden_states: List[Tensor]
    ) -> None:
        """
        :param embedding_output:
            Output of the embedding layer.
        :param layer_hidden_states:
            Outputs of the hidden layers.
        """

        self.all_outputs = [embedding_output]
        self.all_outputs.extend(layer_hidden_states)

    @property
    def embedding_layer(self) -> Tensor:
        """
        Returns the output of the embedding layer.

        :returns:
            Embedding layer output.
            **Shape:** (batch_size,, seq, width)
        """
        return self.all_outputs[0]

    def hidden_layer_states(self, idx: int) -> Tensor:
        """
        Returns the hidden representations of a given layer.

        :param idx:
            Layer index. Must be in ``[0, num_hidden_layers)``.
        :returns:
            Hidden representation of the layer
            **Shape:** (batch_size,, seq, width)
        """
        if 0 <= idx < len(self.all_outputs) - 1:
            return self.all_outputs[idx + 1]
        else:
            raise ValueError(
                "Attempting to select a transformer output tensor using an invalid "
                f"layer index ({idx}). Expected range: 0 <= idx < {(len(self.all_outputs) - 1)}"
            )

    @property
    def last_hidden_layer_states(self) -> Tensor:
        """
        Returns the hidden representation of the last layer.

        :returns:
            Last hidden representation of the last layer.
            **Shape:** (batch_size,, seq, width)
        """
        return self.all_outputs[-1]

    @property
    def all_hidden_layer_states(self) -> List[Tensor]:
        """
        Returns the hidden representation of all the layers.

        :returns:
            Hidden representations of all the layers.
            **Shape:** (batch_size,, seq, width)
        """
        return self.all_outputs[1:]


class ModelOutputWithCache(Generic[CacheT], ModelOutput):
    """
    Output of decoder modules.

    :cache:
        Model cache. The cache can be used with future calls to
        a model to reuse computations for efficiency.
    """

    cache: Optional[List[CacheT]]

    def __init__(
        self,
        *,
        embedding_output: Tensor,
        layer_hidden_states: List[Tensor],
        cache: Optional[List[CacheT]],
    ) -> None:
        """
        :param embedding_output:
            Output of the embedding layer.
        :param layer_hidden_states:
            Outputs of the hidden layers.
        :param cache:
            Model cache.
        """

        super().__init__(
            embedding_output=embedding_output, layer_hidden_states=layer_hidden_states
        )
        self.cache = cache


class CausalLMOutputWithCache(Generic[CacheT], ModelOutputWithCache[CacheT]):
    """
    Output of causal language model modules.

    :logits:
        Logits of the distributions of predicted tokens.
        **Shape:** (batch_size, seq_len, vocab_size)
    """

    logits: Tensor

    def __init__(
        self,
        *,
        embedding_output: Tensor,
        layer_hidden_states: List[Tensor],
        logits: Tensor,
        cache: Optional[List[CacheT]],
    ) -> None:
        """
        :param embedding_output:
            Output of the embedding layer.
        :param layer_hidden_states:
            Outputs of the hidden layers.
        :param logits:
            Logits of the distributions of predicted tokens.
        :param cache:
            Model cache.
        """

        super().__init__(
            cache=cache,
            embedding_output=embedding_output,
            layer_hidden_states=layer_hidden_states,
        )
        self.logits = logits
