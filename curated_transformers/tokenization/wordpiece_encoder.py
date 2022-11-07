from typing import Any, Callable, List, Optional, Tuple
from pathlib import Path

from cutlery import WordPieceProcessor
from spacy.tokens import Doc
from thinc.api import Model, Ragged, deserialize_attr, serialize_attr


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


def build_wordpiece_encoder() -> Model[List[Doc], List[Ragged]]:
    return Model(
        "wordpiece_encoder",
        forward=wordpiece_encoder_forward,
        attrs={
            "wordpiece_processor": WordPieceProcessor([]),
            "unk_piece": "[UNK]",
            "bos_piece": "[CLS]",
            "eos_piece": "[SEP]",
        },
    )


def wordpiece_encoder_forward(
    model: Model, X: List[Doc], is_train: bool
) -> Tuple[List[Ragged], Callable[[Any], Any]]:
    wpp: WordPieceProcessor = model.attrs["wordpiece_processor"]
    bos_piece: str = model.attrs["bos_piece"]
    eos_piece: str = model.attrs["eos_piece"]
    unk_piece: str = model.attrs["unk_piece"]
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
            piece_ids = [
                unk_id if token_id == -1 else token_id
                for token_id in wpp.encode(token.text)[0]
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
    *, path: Path
) -> Callable[
    [Model[List[Doc], List[Ragged]], Optional[List[Doc]], Any],
    Model[List[Doc], List[Ragged]],
]:
    def load(
        model: Model[List[Doc], List[Ragged]], X: List[Doc] = None, Y=None
    ) -> Model[List[Doc], List[Ragged]]:
        model.attrs["wordpiece_processor"] = WordPieceProcessor.from_file(str(path))
        return model

    return load
