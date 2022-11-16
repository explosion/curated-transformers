from typing import List, Optional, TypeVar, Generic
from dataclasses import dataclass
import torch
from torch import Tensor

from thinc.types import Floats2d, Ragged


TrfOutputT = TypeVar("TrfOutputT", Floats2d, Ragged)


@torch.jit.script
class PyTorchTransformerOutput:
    """Output of the PyTorch Transformer Encoders"""

    # The first element is the output of the embedding layer with shape [batch, seq, emb_dim].
    # The rest of the elements are the states of each encoder hidden layer respectively with shape [batch, seq, model_hidden].
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
            raise ValueError(
                f"Index must be >= 0 and < {len(self.all_outputs) - 1}, got {idx}"
            )

    @property
    def last_hidden_layer_states(self) -> Tensor:
        return self.all_outputs[-1]

    @property
    def all_hidden_layer_states(self) -> List[Tensor]:
        return self.all_outputs[1:]


@dataclass
class TransformerModelOutput(Generic[TrfOutputT]):
    """Wrapper for PyTorchTransformerOutput consumed by downstream non-PyTorch components.
    Also acts as the accumulator for the outputs of subsequent models in the Transformer pipeline."""

    # Non-padded, un-stacked versions of the outputs.
    # The outer list tracks Docs and the inner list
    # tracks the embedding + hidden layer outputs of each Doc.
    #
    # The inner-most element is a Floats2d when returned by
    # the PyTorchWrapper transformer model, which are subsequently
    # converted to Ragged by the models that follow.
    all_outputs: List[List[TrfOutputT]]

    # Set to True if only the last hidden layer's outputs are preserved.
    last_layer_only: bool

    def __init__(
        self, *, outputs: List[List[TrfOutputT]], last_layer_only: bool
    ) -> None:
        self.all_outputs = outputs
        self.last_layer_only = last_layer_only

    @property
    def embedding_layers(self) -> List[TrfOutputT]:
        if self.last_layer_only:
            return []
        else:
            return [y[0] for y in self.all_outputs]

    @property
    def last_hidden_layer_states(self) -> List[TrfOutputT]:
        return [y[-1] for y in self.all_outputs]

    @property
    def all_hidden_layer_states(self) -> List[List[TrfOutputT]]:
        return [y[1:] for y in self.all_outputs]

    @property
    def num_outputs(self) -> int:
        return len(self.all_outputs[0])


@dataclass
class DocTransformerOutput:
    """Stored on Doc instances. Each Ragged element corresponds to a layer in
    original TransformerModelOutput, containing piece identifiers."""

    all_outputs: List[Ragged]

    # Set to True if only the last hidden layer's outputs are preserved.
    last_layer_only: bool

    def __init__(self, *, all_outputs: List[Ragged], last_layer_only: bool) -> None:
        self.all_outputs = all_outputs
        self.last_layer_only = last_layer_only

    @property
    def embedding_layer(self) -> Optional[Ragged]:
        if self.last_layer_only:
            return None
        else:
            return self.all_outputs[0]

    @property
    def last_hidden_layer_state(self) -> Ragged:
        return self.all_outputs[-1]

    @property
    def all_hidden_layer_states(self) -> List[Ragged]:
        return self.all_outputs[1:]

    @property
    def num_outputs(self) -> int:
        return len(self.all_outputs)
