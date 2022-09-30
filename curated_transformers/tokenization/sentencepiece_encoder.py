from typing import List, Optional, TypeVar
from cutlery import SentencePieceProcessor
from spacy.tokens import Doc, Span
from thinc.api import Model, Ragged, deserialize_attr, serialize_attr
from thinc.model import empty_init

from .._compat import has_hf_transformers, transformers
from ..util import registry


InT = TypeVar("InT", List[Doc], List[Span])


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


def build_sentencepiece_encoder() -> Model[List[Doc], List[Ragged]]:
    return Model(
        "sentencepiece_encoder",
        forward=sentencepiece_encoder_forward,
        attrs={"sentencepiece_processor": SentencePieceProcessor()},
    )


def sentencepiece_encoder_forward(model: Model, X: InT, is_train: bool):
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


@registry.model_loaders("curated-transformers.HFSentencepieceLoader.v1")
def build_hf_sentencepiece_encoder_loader(*, name, revision: str = "main"):
    def load(model: Model, X: List[Doc] = None, Y=None):
        if not has_hf_transformers:
            raise ValueError("requires 🤗 transformers")

        tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        if not isinstance(tokenizer, transformers.XLMRobertaTokenizerFast):
            raise ValueError("Loading from this 🤗 tokenizer is not supported")

        model.attrs["sentencepiece_processor"] = SentencePieceProcessor.from_file(
            tokenizer.vocab_file
        )

        return model

    return load
