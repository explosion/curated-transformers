from typing import List, Optional, TypeVar, Generic
from dataclasses import dataclass
import torch
from torch import Tensor

from ..errors import Errors


@torch.jit.script
class PyTorchTransformerOutput:
    """Padded output of the PyTorch Transformer encoders."""

    # The first element is the output of the embedding layer with shape [batch, seq, width].
    # The rest of the elements are the states of each encoder hidden layer respectively with shape [batch, seq, width].
    all_outputs: List[Tensor]

    def __init__(
        self, *, embedding_output: Tensor, layer_hidden_states: List[Tensor]
    ) -> None:
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
