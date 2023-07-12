from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import torch
from torch import Tensor

from ..layers.cache import CacheProtocol

CacheT = TypeVar("CacheT", bound=CacheProtocol)


@torch.jit.script
class ModelOutput:
    """
    Base class for model outputs.

    :param all_outputs:
        The first element is the output of the embedding layer. The
        rest of the elements are the states of each encoder hidden
        layer respectively.
    """

    all_outputs: List[Tensor]

    def __init__(
        self, *, embedding_output: Tensor, layer_hidden_states: List[Tensor]
    ) -> None:
        """
        Construct model output.

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

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[0]

    def hidden_layer_states(self, idx: int) -> Tensor:
        """
        Returns the hidden representations of a given layer.

        :param idx:
            Layer index. Must be in ``[0, num_hidden_layers)``.
        :returns:
            Hidden representation of the layer.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        if 0 <= idx < len(self.all_outputs) - 1:
            return self.all_outputs[idx + 1]
        else:
            raise ValueError(
                "Attempting to select a transformer output tensor using an invalid "
                f"layer index ({idx}). Expected range: 0 <= idx < {(len(self.all_outputs) - 1)}"
            )

    @property
    def last_hidden_layer_state(self) -> Tensor:
        """
        Returns the hidden representation of the last layer.

        :returns:
            Last hidden representation of the last layer.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[-1]

    @property
    def all_hidden_layer_states(self) -> List[Tensor]:
        """
        Returns the hidden representation of all the layers.

        :returns:
            Hidden representations of all the layers.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[1:]


class ModelOutputWithCache(Generic[CacheT], ModelOutput):
    """
    Output of decoder modules.

    :param cache:
        Model cache. The cache can be used with future calls
        to a model to reuse computations for efficiency
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
        Construct model output.

        :param embedding_output:
            Output of the embedding layer.

            *Shape:* ``(batch_size, seq_len, width)``
        :param layer_hidden_states:
            Outputs of the hidden layers.

            *Shape:* ``(batch_size, seq_len, width)``
        :param cache:
            Model cache.

            The cache can be used with future calls to
            a model to reuse computations for efficiency
        """

        super().__init__(
            embedding_output=embedding_output, layer_hidden_states=layer_hidden_states
        )
        self.cache = cache


class CausalLMOutputWithCache(Generic[CacheT], ModelOutputWithCache[CacheT]):
    """
    Output of causal language model modules.

    :param logits:
        Logits of the distributions of predicted tokens.
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
        Construct model output.

        :param embedding_output:
            Output of the embedding layer.

            *Shape:* ``(batch_size, seq_len, width)``
        :param layer_hidden_states:
            Outputs of the hidden layers.

            *Shape:* ``(batch_size, seq_len, width)``
        :param logits:
            Logits of the distributions of predicted tokens.

            *Shape:* ``(batch_size, seq_len, vocab_size)``
        :param cache:
            Model cache.
        """

        super().__init__(
            cache=cache,
            embedding_output=embedding_output,
            layer_hidden_states=layer_hidden_states,
        )
        self.logits = logits
