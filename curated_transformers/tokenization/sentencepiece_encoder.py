from typing import Callable, Optional, Tuple
from pathlib import Path
from cutlery import SentencePieceProcessor
from thinc.api import Model, Ragged, chain, deserialize_attr, serialize_attr

from .sentencepiece_adapters import build_camembert_adapter, build_xlmr_adapter

from .types import (
    Tok2PiecesBackpropT,
    Tok2PiecesInT,
    Tok2PiecesModelT,
    Tok2PiecesOutT,
)


@serialize_attr.register(SentencePieceProcessor)
def serialize_sentencepiece_processor(
    _, value: SentencePieceProcessor, name: str, model
) -> bytes:
    return value.to_protobuf()


@deserialize_attr.register(SentencePieceProcessor)
def deserialize_my_custom_class(
    _, value: bytes, name: str, model
) -> SentencePieceProcessor:
    return SentencePieceProcessor.from_protobuf(value)


def build_camembert_sentencepiece_encoder_v1() -> Tok2PiecesModelT:
    """Construct a SentencePiece piece encoder model that accepts a list
    of token sequences or documents and returns a corresponding list
    of piece identifiers with CamemBERT post-processing applied.

    This model must be separately initialized using an appropriate
    loader.
    """
    encoder = build_sentencepiece_encoder_v1()
    model = chain(encoder, build_camembert_adapter())
    model.set_ref("encoder", encoder)
    return model


def build_sentencepiece_encoder_v1() -> Tok2PiecesModelT:
    """Construct a SentencePiece piece encoder model that accepts a list
    of token sequences or documents and returns a corresponding list
    of piece identifiers.

    This model must be separately initialized using an appropriate
    loader.
    """
    model: Tok2PiecesModelT = Model(
        "sentencepiece_encoder",
        forward=sentencepiece_encoder_forward,
        attrs={"sentencepiece_processor": SentencePieceProcessor()},
    )
    model.set_ref("encoder", model)
    return model


def build_xlmr_sentencepiece_encoder_v1() -> Tok2PiecesModelT:
    """Construct a SentencePiece piece encoder model that accepts a list
    of token sequences or documents and returns a corresponding list
    of piece identifiers with XLM-RoBERTa post-processing applied.

    This model must be separately initialized using an appropriate
    loader.
    """
    encoder = build_sentencepiece_encoder_v1()
    model = chain(encoder, build_xlmr_adapter())
    model.set_ref("encoder", encoder)
    return model


def sentencepiece_encoder_forward(
    model: Model, X: Tok2PiecesInT, is_train: bool
) -> Tuple[Tok2PiecesOutT, Tok2PiecesBackpropT]:
    spp: SentencePieceProcessor = model.attrs["sentencepiece_processor"]

    pieces = []
    for doc in X:
        # TODO: check whether as single bos/eos per doc is what we want.
        #   The issue is that we probably do not have sentence
        #   boundaries yet, since they are predicted by a pipe.
        doc_pieces = [spp.bos_id()]
        lens = [1]

        for token in doc:
            piece_ids = spp.encode_as_ids(token.text)
            doc_pieces.extend(piece_ids)
            lens.append(len(piece_ids))

        doc_pieces.append(spp.eos_id())
        lens.append(1)
        pieces.append(
            Ragged(
                model.ops.asarray1i(doc_pieces),
                model.ops.asarray1i(lens),
            )
        )

    return pieces, lambda dY: []


def build_sentencepiece_encoder_loader_v1(
    *, path: Path
) -> Callable[
    [Tok2PiecesModelT, Optional[Tok2PiecesInT], Optional[Tok2PiecesInT]],
    Tok2PiecesModelT,
]:
    """Construct a callback that initializes a SentencePiece piece encoder
    model.

    path (Path):
        Path to the serialized SentencePiece model.
    """

    def load(model, X=None, Y=None):
        model.attrs["sentencepiece_processor"] = SentencePieceProcessor.from_file(
            str(path)
        )
        return model

    return load
