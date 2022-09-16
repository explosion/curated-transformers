from typing import List, Optional, Tuple

from spacy.tokens import Span, Doc
from thinc.layers import chain, Embed, with_array, with_padded
from thinc.model import Model
from thinc.types import Ragged, ArrayXd

from ..tokenization.sentencepiece_adapters import build_xlmr_adapter, remove_bos_eos
from ..tokenization.sentencepiece_encoder import build_hf_sentencepiece_encoder
from ..tokenization.sentencepiece_encoder import build_sentencepiece_encoder
from ..tokenization.wordpiece_encoder import build_hf_wordpiece_encoder
from ..tokenization.wordpiece_encoder import build_wordpiece_encoder
from .hf_wrapper import (
    build_hf_transformer_encoder_v1,
    bert_encoder_from_pretrained_hf_model,
)
from .hf_wrapper import roberta_encoder_from_pretrained_hf_model


def build_bert_transformer_model_v1(
    *, with_spans, hf_model_name: Optional[str] = None, hf_model_revision: str = "main"
):
    if not hf_model_name:
        piece_encoder = build_wordpiece_encoder()
        transformer = with_array(_stubformer(768, 1000))
    else:
        piece_encoder = build_hf_wordpiece_encoder(
            hf_model_name=hf_model_name, hf_model_revision=hf_model_revision
        )
        encoder = bert_encoder_from_pretrained_hf_model(
            model_name=hf_model_name, model_revision=hf_model_revision
        )
        transformer = build_hf_transformer_encoder_v1(encoder)

    return build_transformer_model_v1(
        with_spans=with_spans,
        piece_encoder=piece_encoder,
        transformer=transformer,
    )


def build_xlmr_transformer_model_v1(
    *, with_spans, hf_model_name: Optional[str] = None, hf_model_revision: str = "main"
):
    piece_adapter = build_xlmr_adapter()

    if not hf_model_name:
        piece_encoder = build_sentencepiece_encoder()
        transformer = with_array(_stubformer(768, 1000))
    else:
        piece_encoder = build_hf_sentencepiece_encoder(hf_model_name)
        encoder = roberta_encoder_from_pretrained_hf_model(
            model_name=hf_model_name, model_revision=hf_model_revision
        )
        transformer = build_hf_transformer_encoder_v1(encoder)

    return build_transformer_model_v1(
        with_spans=with_spans,
        piece_encoder=piece_encoder,
        piece_adapter=piece_adapter,
        transformer=transformer,
    )


def build_transformer_model_v1(
    *,
    with_spans,
    piece_encoder: Model[List[Span], List[Ragged]],
    piece_adapter: Model[List[Ragged], List[Ragged]] = None,
    transformer: Model[List[Ragged], List[Ragged]],
):
    # FIXME: do we want to make `remove_bos_eos` configurable as well or
    #        is it always the same post-processing?
    if piece_adapter:
        layers = [
            chain(
                piece_encoder, piece_adapter, with_spans(transformer), remove_bos_eos()
            )
        ]
    else:
        layers = [chain(piece_encoder, with_spans(transformer), remove_bos_eos())]
    refs = {
        "piece_encoder": piece_encoder,
        "transformer": transformer,
    }
    return Model(
        "transformer_model",
        transformer_model_forward,
        init=transformer_model_init,
        layers=layers,
        refs=refs,
    )


def transformer_model_forward(model: Model, docs: List[Doc], is_train: bool):
    Y, backprop_layer = model.layers[0](docs, is_train=is_train)

    def backprop(dY):
        backprop_layer(dY)

        # Return empty list for backprop, since we cannot backprop into piece
        # identifiers.
        return []

    return Y, backprop


def transformer_model_init(model: Model, X: List[Doc] = None, Y=None):
    model.layers[0].initialize(X, Y)
    return model


# Not a real transformer, just a stub.
def _stubformer(nO, nV):
    return Embed(nO, nV)
