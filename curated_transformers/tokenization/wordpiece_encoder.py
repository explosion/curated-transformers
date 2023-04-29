from typing import Callable, List, Optional, Tuple
from pathlib import Path
import unicodedata

from cutlery import WordPieceProcessor
from thinc.api import Model, Ragged, deserialize_attr, serialize_attr

from .types import (
    Tok2PiecesBackpropT,
    Tok2PiecesInT,
    Tok2PiecesModelT,
    Tok2PiecesOutT,
)


@serialize_attr.register(WordPieceProcessor)
def serialize_sentencepiece_processor(
    _, value: WordPieceProcessor, name: str, model
) -> bytes:
    return "\n".join(value.to_list()).encode("utf8")


@deserialize_attr.register(WordPieceProcessor)
def deserialize_my_custom_class(
    _, value: bytes, name: str, model
) -> WordPieceProcessor:
    return WordPieceProcessor(value.decode("utf8").split("\n"))


def build_bert_wordpiece_encoder_v1() -> Tok2PiecesModelT:
    """Construct a WordPiece piece encoder model that accepts a list
    of token sequences or documents and returns a corresponding list
    of piece identifiers. This encoder also splits each token
    on punctuation characters, as expected by most BERT models.

    This model must be separately initialized using an appropriate
    loader.
    """
    return Model(
        "wordpiece_encoder",
        forward=wordpiece_encoder_forward,
        attrs={
            "wordpiece_processor": WordPieceProcessor([]),
            "unk_piece": "[UNK]",
            "bos_piece": "[CLS]",
            "eos_piece": "[SEP]",
            "lowercase": False,
            "preprocess": _bert_preprocess,
            "strip_accents": False,
        },
    )


def build_wordpiece_encoder_v1() -> Tok2PiecesModelT:
    """Construct a WordPiece piece encoder model that accepts a list
    of token sequences or documents and returns a corresponding list
    of piece identifiers.

    This model must be separately initialized using an appropriate
    loader.
    """
    return Model(
        "wordpiece_encoder",
        forward=wordpiece_encoder_forward,
        attrs={
            "wordpiece_processor": WordPieceProcessor([]),
            "unk_piece": "[UNK]",
            "bos_piece": "[CLS]",
            "eos_piece": "[SEP]",
            "lowercase": False,
            "preprocess": lambda t: [t],
            "strip_accents": False,
        },
    )


def wordpiece_encoder_forward(
    model: Model, X: Tok2PiecesInT, is_train: bool
) -> Tuple[Tok2PiecesOutT, Tok2PiecesBackpropT]:
    wpp: WordPieceProcessor = model.attrs["wordpiece_processor"]
    bos_piece: str = model.attrs["bos_piece"]
    eos_piece: str = model.attrs["eos_piece"]
    unk_piece: str = model.attrs["unk_piece"]
    lowercase: bool = model.attrs["lowercase"]
    preprocess: Callable[[str], str] = model.attrs["preprocess"]
    strip_accents: bool = model.attrs["strip_accents"]
    bos_id = wpp.get_initial(bos_piece)
    eos_id = wpp.get_initial(eos_piece)
    unk_id = wpp.get_initial(unk_piece)

    pieces = []
    for doc in X:
        # TODO: check whether as single bos/eos per doc is what we want.
        #   The issue is that we probably do not have sentence
        #   boundaries yet, since they are predicted by a pipe.
        doc_pieces = [bos_id]
        lens = [1]

        for token in doc:
            text = token.lower_ if lowercase else token.text
            if strip_accents:
                text = _strip_accents(text)

            piece_ids = [
                unk_id if token_id == -1 else token_id
                for t in preprocess(text)
                for token_id in wpp.encode(t)[0]
            ]

            doc_pieces.extend(piece_ids)
            lens.append(len(piece_ids))

        doc_pieces.append(eos_id)
        lens.append(1)
        pieces.append(
            Ragged(
                model.ops.asarray1i(doc_pieces),
                model.ops.asarray1i(lens),
            )
        )

    return pieces, lambda dY: []


def build_wordpiece_encoder_loader_v1(
    *,
    path: Path,
    bos_piece="[CLS]",
    eos_piece="[SEP]",
    unk_piece="[UNK]",
    lowercase: bool = False,
    strip_accents: bool = False,
) -> Callable[
    [Tok2PiecesModelT, Optional[Tok2PiecesInT], Optional[Tok2PiecesInT]],
    Tok2PiecesModelT,
]:
    """Construct a callback that initializes a WordPiece piece encoder
    model.

    path (Path):
        Path to the serialized WordPiece model.
    """

    def load(model, X=None, Y=None):
        model.attrs["wordpiece_processor"] = WordPieceProcessor.from_file(str(path))
        model.attrs["bos_piece"] = bos_piece
        model.attrs["eos_piece"] = eos_piece
        model.attrs["unk_piece"] = unk_piece
        model.attrs["lowercase"] = lowercase
        model.attrs["strip_accents"] = strip_accents
        return model

    return load


def _bert_preprocess(token: str) -> List[str]:
    """Split a token on punctuation characters. For instance,
    'AWO-Mitarbeiter' is split into ['AWO', '-', 'Mitarbeiter']"""
    tokens = []
    in_word = False
    while token:
        char = token[0]
        token = token[1:]
        if _is_bert_punctuation(char):
            tokens.append([char])
            in_word = False
        else:
            if in_word:
                tokens[-1].append(char)
            else:
                tokens.append([char])
                in_word = True
    return ["".join(t) for t in tokens]


def _is_bert_punctuation(char: str) -> bool:
    """Checks whether `char` is a punctuation character."""
    # ASCII punctuation from HF tranformers, since we need to split
    # in the same way.
    cp = ord(char)
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True

    return unicodedata.category(char).startswith("P")


def _strip_accents(token: str) -> str:
    token = unicodedata.normalize("NFD", token)
    return "".join([char for char in token if unicodedata.category(char) != "Mn"])
