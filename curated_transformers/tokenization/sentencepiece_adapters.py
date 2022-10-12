from typing import List, Union

from thinc.api import Model, Ragged

from ..models.output import TransformerModelOutput

_FAIRSEQ_OFFSET = 1
_FAIRSEQ_BOS = 0
_FAIRSEQ_EOS = 2
_FAIRSEQ_UNK = 3

_SPP_BOS = 1
_SPP_EOS = 2
_SPP_UNK = 0

RemoveBosEosBackpropInOutT = Union[List[Ragged], List[List[Ragged]]]


def _update_to_fairseq(piece_id):
    if piece_id == _SPP_UNK:
        return _FAIRSEQ_UNK
    elif piece_id == _SPP_BOS:
        return _FAIRSEQ_BOS
    elif piece_id == _SPP_EOS:
        return _FAIRSEQ_EOS
    else:
        return piece_id + _FAIRSEQ_OFFSET


def _update_to_fairseq_vectorized(xp):
    return xp.vectorize(_update_to_fairseq)


def build_xlmr_adapter() -> Model[List[Ragged], List[Ragged]]:
    return Model(
        "sentencepiece_encoder",
        forward=xlmr_adapter_forward,
    )


def xlmr_adapter_forward(model: Model, X: List[Ragged], is_train: bool):
    # Align original fairseq vocab with the sentencepiece vocabulary.
    update_to_fairseq = _update_to_fairseq_vectorized(model.ops.xp)
    X_xlmr = []
    for tokens_pieces in X:
        X_xlmr.append(
            Ragged(
                data=update_to_fairseq(tokens_pieces.dataXd),
                lengths=tokens_pieces.lengths,
            )
        )

    return X_xlmr, lambda dY: []


def remove_bos_eos() -> Model[TransformerModelOutput, TransformerModelOutput]:
    return Model("remove_bos_eos", remove_bos_eos_forward)


def remove_bos_eos_forward(model: Model, X: TransformerModelOutput, is_train: bool):
    if not isinstance(X, TransformerModelOutput):
        raise ValueError(f"Unsupported input of type '{type(X)}'")

    X.all_outputs = [[Xr[1:-1] for Xr in inner] for inner in X.all_outputs]

    def backprop(dY: RemoveBosEosBackpropInOutT):
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
