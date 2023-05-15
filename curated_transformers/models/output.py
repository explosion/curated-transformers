from typing import Generic, List, Optional, TypeVar
import torch
from torch import Tensor


CacheT = TypeVar("CacheT")


@torch.jit.script
class ModelOutput:
    """Padded output of the PyTorch Transformer encoders."""

    # The first element is the output of the embedding layer with shape [batch, seq, width].
    # The rest of the elements are the states of each encoder hidden layer respectively with shape [batch, seq, width].
    all_outputs: List[Tensor]

    def __init__(
        self, *, embedding_output: Tensor, layer_hidden_states: List[Tensor]
    ) -> None:
        """
        :param embedding_output: Output of the embedding layer.
        :param layer_hidden_states: Outputs of the hidden layers.
        """

        self.all_outputs = [embedding_output]
        self.all_outputs.extend(layer_hidden_states)

    @property
    def embedding_layer(self) -> Tensor:
        return self.all_outputs[0]

    def hidden_layer_states(self, idx: int) -> Tensor:
        """'idx' must be in the range [0, num_hidden_layers)"""
        if 0 <= idx < len(self.all_outputs) - 1:
            return self.all_outputs[idx + 1]
        else:
            # This error needs to be inlined as due to torch.jit.script limitations.
            raise ValueError(
                "Attempting to select a transformer output tensor using an invalid "
                f"layer index ({idx}). Expected range: 0<= idx < {(len(self.all_outputs) - 1)}"
            )

    @property
    def last_hidden_layer_states(self) -> Tensor:
        return self.all_outputs[-1]

    @property
    def all_hidden_layer_states(self) -> List[Tensor]:
        return self.all_outputs[1:]


class ModelOutputWithCache(Generic[CacheT], ModelOutput):
    """Output of causal language models.

    :cache: Model cache. The cache can be used by future calls to
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
        :param embedding_output: Output of the embedding layer.
        :param layer_hidden_states: Outputs of the hidden layers.
        :param cache: Model cache.
        """

        super().__init__(
            embedding_output=embedding_output, layer_hidden_states=layer_hidden_states
        )
        self.cache = cache


class CausalLMOutputWithCache(Generic[CacheT], ModelOutputWithCache[CacheT]):
    """
    Output of causal language models.

    :logits: Logits of the distributions of predicted tokens.
    """

    # Shape: [batch_size, seq_len, vocab_size]
    logits: Tensor

    def __init__(
        self,
        *,
        cache: Optional[List[CacheT]],
        embedding_output: Tensor,
        layer_hidden_states: List[Tensor],
        logits: Tensor,
    ) -> None:
        """
        :param embedding_output: Output of the embedding layer.
        :param layer_hidden_states: Outputs of the hidden layers.
        :param cache: Model cache.
        :param logits: Logits of the distributions of predicted tokens.
        """

        super().__init__(
            cache=cache,
            embedding_output=embedding_output,
            layer_hidden_states=layer_hidden_states,
        )
        self.logits = logits
