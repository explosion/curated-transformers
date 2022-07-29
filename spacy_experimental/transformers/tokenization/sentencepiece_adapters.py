from typing import List
from thinc.api import Model, Ragged

_FAIRSEQ_OFFSET = 1
_FAIRSEQ_BOS = 0
_FAIRSEQ_EOS = 2
_FAIRSEQ_UNK = 3

_SPP_BOS = 1
_SPP_EOS = 2
_SPP_UNK = 0


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


def remove_bos_eos() -> Model[List[Ragged], List[Ragged]]:
    return Model("remote_bos_eos", remove_bos_eos_forward)


def remove_bos_eos_forward(model: Model, X: List[Ragged], is_train: bool):
    X_removed = [Ragged(Xr.data[1:-1], lengths=Xr.lengths[1:-1]) for Xr in X]

    def backprop(dY: List[Ragged]):
        # Pass-through dY, but add zero gradient for the special bos/eos
        # tokens.
        dX = []
        for dYr in dY:
            dim0 = dYr.dataXd.shape[0] + 2

            data = model.ops.alloc_f((dim0,) + dYr.dataXd.shape[1:])
            data[[0, -1]] = 0.0
            data[1:-1] = dYr.dataXd

            lengths = model.ops.alloc1i(dYr.lengths.shape[0] + 2)
            lengths[[0, 1]] = 1
            lengths[1:-1] = dYr.lengths

            dX.append(Ragged(data, lengths=lengths))

        return dX

    return X_removed, backprop
