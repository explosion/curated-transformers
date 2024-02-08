from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from torch import Tensor

from ..layers.cache import CacheProtocol

CacheT = TypeVar("CacheT", bound=CacheProtocol)


@dataclass
class ModelOutput:
    """
    Base class for model outputs.

    :param all_outputs:
        The first element is the output of the embedding layer. The
        rest of the elements are the states of each encoder hidden
        layer respectively.
    """

    all_outputs: List[Tensor]

    @property
    def embedding_layer(self) -> Tensor:
        """
        Return the output of the embedding layer.

        :returns:
            Embedding layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[0]

    def hidden_layer_states(self, idx: int) -> Tensor:
        """
        Return the hidden representations of a given layer.

        :param idx:
            Layer index. Must be in ``[0, n_hidden_layers)``.
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
        Return the hidden representation of the last layer.

        :returns:
            Last hidden representation of the last layer.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[-1]

    @property
    def all_hidden_layer_states(self) -> List[Tensor]:
        """
        Return the hidden representation of all the layers.

        :returns:
            Hidden representations of all the layers.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[1:]


@dataclass
class ModelOutputWithCache(Generic[CacheT], ModelOutput):
    """
    Output of decoder modules.

    :param cache:
        Model cache. The cache can be used with future calls
        to a model to reuse computations for efficiency
    """

    cache: Optional[List[CacheT]]


@dataclass
class CausalLMOutputWithCache(Generic[CacheT], ModelOutputWithCache[CacheT]):
    """
    Output of causal language model modules.

    :param logits:
        Logits of the distributions of predicted tokens.
    """

    logits: Tensor
