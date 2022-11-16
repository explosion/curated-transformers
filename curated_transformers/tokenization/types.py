from typing import Any, Callable, List, TypeVar

from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from thinc.model import Model
from thinc.types import Ragged


TokenSequenceT = TypeVar("TokenSequenceT", Doc, List[Token])

Tok2PiecesInT = List[TokenSequenceT]
Tok2PiecesOutT = List[Ragged]
Tok2PiecesBackpropT = Callable[[Any], Any]
Tok2PiecesModelT = Model[Tok2PiecesInT, Tok2PiecesOutT]

PieceAdapterInOutT = List[Ragged]
PieceAdapterBackpropT = Callable[[Any], Any]
PieceAdapterModelT = Model[PieceAdapterInOutT, PieceAdapterInOutT]
