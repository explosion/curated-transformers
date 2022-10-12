from typing import List, TypeVar
from dataclasses import dataclass
import torch
from torch import Tensor

from thinc.types import Floats2d, Ragged


TrfOutputT = TypeVar("TrfOutputT", Floats2d, Ragged)


@torch.jit.script
class PyTorchTransformerOutput:
    """Output of the PyTorch Transformer Encoders"""

    # The first element is the output of the embedding layer with shape [batch, seq, emb_dim].
    # The rest of the elements are the hidden states of each encoder layer respectively with shape [batch, seq, model_hidden].
    all_outputs: List[Tensor]

    def __init__(
        self, *, embedding_output: Tensor, layer_hidden_states: List[Tensor]
    ) -> None:
        self.all_outputs = [embedding_output]
        self.all_outputs.extend(layer_hidden_states)

    @property
    def embedding_output(self) -> Tensor:
        return self.all_outputs[0]

    def layer_hidden_state(self, idx: int) -> Tensor:
        """'idx' must be in the range [0, num_hidden_layers)"""
        if 0 <= idx < len(self.all_outputs) - 1:
            return self.all_outputs[idx + 1]
        else:
            raise ValueError(
                f"Index must be >= 0 and < {len(self.all_outputs) - 1}, got {idx}"
            )

    @property
    def last_hidden_state(self) -> Tensor:
        return self.all_outputs[-1]

    @property
    def all_layer_hidden_states(self) -> List[Tensor]:
        return self.all_outputs[1:]


@dataclass
class TransformerModelOutput:
    """Wrapper for PyTorchTransformerOutput consumed by downstream non-PyTorch components.
    Also acts as the accumulator for the outputs of subsequent models in the Transformer pipeline."""

    # Non-padded, un-stacked versions of the outputs.
    # The outer list tracks Docs/spans and the inner
    # list tracks the embedding + layer outputs of each Doc/span.
    #
    # The inner-most element is a Floats2d when returned by
    # the PyTorchWrapper transformer model, which are subsequently
    # converted to Ragged by the models that follow.
    all_outputs: List[List[TrfOutputT]]

    # Set to True if only the last layer's outputs are preserved.
    last_layer_only: bool

    def __init__(self, *, outputs: List[List[TrfOutputT]]) -> None:
        self.all_outputs = outputs

    @property
    def embedding_outputs(self) -> List[TrfOutputT]:
        return [y[0] for y in self.all_outputs]

    @property
    def last_hidden_states(self) -> List[TrfOutputT]:
        return [y[-1] for y in self.all_outputs]

    @property
    def all_layer_hidden_states(self) -> List[List[TrfOutputT]]:
        return [y[1:] for y in self.all_outputs]

    @property
    def num_outputs(self) -> int:
        return len(self.all_outputs[0])


@dataclass
class DocTransformerOutput:
    """Stored on Doc instances. Each Ragged element corresponds to a layer in
    original TransformerModelOutput, containing piece identifiers."""

    layer_outputs: List[Ragged]

    # Set to True if only the last layer's outputs are preserved.
    last_layer_only: bool

    def __init__(self, *, layer_outputs: List[Ragged]) -> None:
        self.layer_outputs = layer_outputs

    @property
    def last_hidden_state(self) -> Ragged:
        return self.layer_outputs[-1]
