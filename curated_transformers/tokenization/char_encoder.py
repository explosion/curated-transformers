from typing import Callable, Dict, Optional, OrderedDict, Tuple
from pathlib import Path
from thinc.api import Model, Ragged
import unicodedata

from .types import Tok2PiecesBackpropT, Tok2PiecesInT, Tok2PiecesModelT, Tok2PiecesOutT
from ..errors import Errors


def build_char_encoder_v1() -> Tok2PiecesModelT:
    """Construct a character piece encoder model that accepts a list
    of token sequences or documents and returns a corresponding list
    of piece identifiers.

    This model must be separately initialized using an appropriate
    loader.

    normalize (str):
       Unicode normalization to apply before encoding a token. Defaults to
       "NFKC".
    """
    return Model(
        "char_encoder",
        forward=char_encoder_forward,
        attrs={
            "bos_piece": "[CLS]",
            "eos_piece": "[SEP]",
            "unk_piece": "[UNK]",
            "normalize": "NFKC",
            "vocab": None,
        },
    )


def char_encoder_forward(
    model: Model, X: Tok2PiecesInT, is_train: bool
) -> Tuple[Tok2PiecesOutT, Tok2PiecesBackpropT]:
    """Construct a character piece encoder model that accepts a list
    of token sequences or documents and returns a corresponding list
    of piece identifiers.

    This model must be separately initialized using an appropriate
    loader.
    """
    vocab: Optional[Dict[str, int]] = model.attrs["vocab"]
    if vocab is None:
        raise ValueError(Errors.E020)

    bos_piece: str = model.attrs["bos_piece"]
    eos_piece: str = model.attrs["eos_piece"]
    unk_piece: str = model.attrs["unk_piece"]
    normalize: Optional[str] = model.attrs["normalize"]
    bos_id = vocab[bos_piece]
    eos_id = vocab[eos_piece]
    unk_id = vocab[unk_piece]

    pieces = []
    for doc in X:
        doc_pieces = [bos_id]
        lens = [1]

        for token in doc:
            text = (
                unicodedata.normalize(normalize, token.text)
                if normalize is not None
                else token.text
            )
            # Most encoders will mark a full token as unknown. The character
            # encoder behaves differently, only replacing unknown characters
            # by the unk id.
            piece_ids = [vocab.get(char, unk_id) for char in text]
            doc_pieces.extend(piece_ids)
            lens.append(len(piece_ids))

        doc_pieces.append(eos_id)
        lens.append(1)
        pieces.append(
            Ragged(model.ops.asarray1i(doc_pieces), model.ops.asarray1i(lens))
        )

    return pieces, lambda dY: []


def build_char_encoder_loader_v1(
    *,
    path: Path,
    bos_piece: str = "[CLS]",
    eos_piece: str = "[SEP]",
    unk_piece: str = "[UNK]",
    normalize: Optional[str] = "NFKC",
) -> Callable[
    [Tok2PiecesModelT, Optional[Tok2PiecesInT], Optional[Tok2PiecesInT]],
    Tok2PiecesModelT,
]:
    """Construct a callback that initializes a character piece encoder
    model.

    path (Path):
        Path to the serialized character model.
    """

    def load(model, X=None, Y=None):
        if model.name != "char_encoder":
            raise ValueError(Errors.E021.format(model_name=model.name))

        model.attrs["bos_piece"] = bos_piece
        model.attrs["eos_piece"] = eos_piece
        model.attrs["unk_piece"] = unk_piece
        model.attrs["normalize"] = normalize

        vocab = OrderedDict()
        with open(path, encoding="utf-8") as f:
            for char in f:
                char = char.rstrip("\r\n")
                if normalize is not None:
                    char = unicodedata.normalize(normalize, char)
                vocab[char] = len(vocab)
        model.attrs["vocab"] = vocab
        return model

    return load
