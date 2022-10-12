from typing import List, Optional, TypeVar

from cutlery import WordPieceProcessor
from spacy.tokens import Doc, Span
from thinc.api import Model, Ragged, deserialize_attr, serialize_attr
from thinc.model import empty_init

from .._compat import has_hf_transformers, transformers
from ..util import registry

InT = TypeVar("InT", List[Doc], List[Span])


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


@registry.model_loaders("curated-transformers.HFWordpieceLoader.v1")
def build_hf_wordpiece_encoder_loader(*, name, revision: str = "main"):
    def load(model: Model, X: List[Doc] = None, Y=None):
        if not has_hf_transformers:
            raise ValueError("requires 🤗 transformers")

        tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        if not isinstance(tokenizer, transformers.BertTokenizerFast):
            raise ValueError("Loading from this 🤗 tokenizer is not supported")

        # Seems like we cannot get the vocab file name for a BERT vocabulary? So,
        # instead, copy the vocabulary.
        vocab = [None] * tokenizer.vocab_size
        for piece, idx in tokenizer.vocab.items():
            vocab[idx] = piece
        model.attrs["wordpiece_processor"] = WordPieceProcessor(vocab)
        model.attrs["bos_piece"] = tokenizer.cls_token
        model.attrs["eos_piece"] = tokenizer.sep_token
        model.attrs["unk_piece"] = tokenizer.unk_token

    return load


def wordpiece_encoder_forward(model: Model, X: InT, is_train: bool):
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
