from typing import List, Tuple
from thinc.api import Model, Ragged

from .output import TransformerModelOutput
from .types import (
    SentMarkerRemoverInOutT,
    SentMarkerRemoverBackpropT,
    SentMarkerRemoverModelT,
    RaggedInOutT,
)


def remove_bos_eos() -> SentMarkerRemoverModelT:
    return Model("remove_bos_eos", remove_bos_eos_forward)


def remove_bos_eos_forward(
    model: Model, X: SentMarkerRemoverInOutT, is_train: bool
) -> Tuple[SentMarkerRemoverInOutT, SentMarkerRemoverBackpropT]:
    if not isinstance(X, TransformerModelOutput):
        raise ValueError(f"Unsupported input of type '{type(X)}'")

    X.all_outputs = [[Xr[1:-1] for Xr in inner] for inner in X.all_outputs]

    def backprop(dY: RaggedInOutT) -> RaggedInOutT:
        # Pass-through dY, but add zero gradient for the special bos/eos
        # tokens.

        def _apply_to_layer(input: List[Ragged], output: List[Ragged]):
            for dYr in input:
                dim0 = dYr.dataXd.shape[0] + 2

                data = model.ops.xp.empty((dim0,) + dYr.dataXd.shape[1:], dtype="f")
                data[[0, -1]] = 0.0
                data[1:-1] = dYr.dataXd

                lengths = model.ops.alloc1i(dYr.lengths.shape[0] + 2, zeros=False)
                lengths[[0, -1]] = 1
                lengths[1:-1] = dYr.lengths
                output.append(Ragged(data, lengths=lengths))

        def _apply_to_layers(input: List[List[Ragged]], output: List[List[Ragged]]):
            for inner_dY in input:
                inner_dX = []
                _apply_to_layer(inner_dY, inner_dX)
                output.append(inner_dX)

        dX = []
        if isinstance(dY[0], list):
            _apply_to_layers(dY, dX)
        else:
            _apply_to_layer(dY, dX)

        return dX

    return X, backprop
