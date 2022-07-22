from typing import List
from thinc.api import Model, Ragged

_FAIRSEQ_OFFSET = 1
_FAIRSEQ_UNK = 3
_SPP_UNK = 0


def _update_to_fairseq(piece_id):
    # TODO: add bos/eos pieces.
    if piece_id == _SPP_UNK:
        return _FAIRSEQ_UNK
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
