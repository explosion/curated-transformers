from typing import Tuple
from functools import lru_cache

from thinc.api import Model, Ragged

from .types import (
    PieceAdapterBackpropT,
    PieceAdapterInOutT,
    PieceAdapterModelT,
)

_FAIRSEQ_OFFSET = 1
_CAMEMBERT_FAIRSEQ_OFFSET = 4
_FAIRSEQ_BOS = 0
_FAIRSEQ_EOS = 2
_FAIRSEQ_UNK = 3

_SPP_BOS = 1
_SPP_EOS = 2
_SPP_UNK = 0


def _update_to_fairseq(piece_id: int):
    if piece_id == _SPP_UNK:
        return _FAIRSEQ_UNK
    elif piece_id == _SPP_BOS:
        return _FAIRSEQ_BOS
    elif piece_id == _SPP_EOS:
        return _FAIRSEQ_EOS
    else:
        return piece_id + _FAIRSEQ_OFFSET


@lru_cache(maxsize=128)
def _update_to_fairseq_vectorized(xp):
    return xp.vectorize(_update_to_fairseq)


def build_xlmr_adapter() -> PieceAdapterModelT:
    """Align the original fairseq vocab used by pre-trained HF transformer
    models with the sentencepiece vocabulary."""

    return Model(
        "xlmr_adapter",
        forward=xlmr_adapter_forward,
    )


def xlmr_adapter_forward(
    model: Model, X: PieceAdapterInOutT, is_train: bool
) -> Tuple[PieceAdapterInOutT, PieceAdapterBackpropT]:
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


def _camembert_update_to_fairseq(piece_id: int) -> int:
    if piece_id == _SPP_UNK:
        return _FAIRSEQ_UNK
    else:
        return piece_id + _CAMEMBERT_FAIRSEQ_OFFSET


@lru_cache(maxsize=128)
def _camembert_update_to_fairseq_vectorized(xp):
    return xp.vectorize(_camembert_update_to_fairseq)


def build_camembert_adapter() -> PieceAdapterModelT:
    """Align the original fairseq vocab used by pre-trained Camembert
    HF transformer model with the sentencepiece vocabulary."""
    return Model(
        "camembert_adapter",
        forward=camembert_adapter_forward,
    )


def camembert_adapter_forward(
    model: Model, X: PieceAdapterInOutT, is_train: bool
) -> Tuple[PieceAdapterInOutT, PieceAdapterBackpropT]:
    update_to_fairseq = _camembert_update_to_fairseq_vectorized(model.ops.xp)
    X_xlmr = []
    for tokens_pieces in X:
        X_xlmr.append(
            Ragged(
                data=update_to_fairseq(tokens_pieces.dataXd),
                lengths=tokens_pieces.lengths,
            )
        )

    return X_xlmr, lambda dY: []
