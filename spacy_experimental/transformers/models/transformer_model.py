from typing import List, Tuple

from spacy.tokens import Span, Doc
from thinc.layers import chain, Embed, with_array
from thinc.model import Model
from thinc.types import Ragged, ArrayXd

from ..tokenization.sentencepiece_adapters import build_xlmr_adapter, remove_bos_eos
from ..tokenization.sentencepiece_encoder import build_sentencepiece_encoder


def build_xlmr_transformer_model_v1(*, with_spans):
    piece_encoder = build_sentencepiece_encoder()
    piece_adapter = build_xlmr_adapter()
    return build_transformer_model_v1(
        with_spans=with_spans,
        piece_encoder=piece_encoder,
        piece_adapter=piece_adapter,
        transformer=with_array(_stubformer(768, 1000)),
    )


def build_transformer_model_v1(
    *,
    with_spans,
    piece_encoder: Model[List[Span], List[Ragged]],
    piece_adapter: Model[List[Ragged], List[Ragged]],
    transformer: Model[List[Ragged], List[Ragged]],
):
    # FIXME: do we want to make `remove_bos_eos` configurable as well or
    #        is it always the same post-processing?
    layers = [
        chain(piece_encoder, piece_adapter, with_spans(transformer), remove_bos_eos())
    ]
    refs = {
        "piece_encoder": piece_encoder,
        "tokenizer": layers[0],
        "transformer": with_spans(transformer),
        "remove_bos_eos": layers[-1],
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


# Only needed for wrapping the stubformer.
def _with_array_from_list_ragged(layer: Model):
    def init(model, X, Y):
        if X:
            Xlf = layer.ops.flatten([r.dataXd for r in X])
            layer.initialize(Xlf, Y)

        return model

    def forward(model, Xlr, is_train):
        Xla, lens = _raggeds_to_arrays(Xlr)
        Xf = layer.ops.flatten(Xla)
        Yf, backprop_layer = layer(Xf, is_train=is_train)

        def backprop(dY):
            dYla, lens = _raggeds_to_arrays(dY)
            dYf = layer.ops.flatten(dYla)
            dXf = backprop_layer(dYf)
            dXla = layer.ops.unflatten(dXf, lengths=lens)
            return [Ragged(grads, xr.lengths) for grads, xr in zip(dXla, Xlr)]

        Y_arr = layer.ops.unflatten(Yf, lengths=lens)
        Ylr = [Ragged(ya, xr.lengths) for ya, xr in zip(Y_arr, Xlr)]

        return Ylr, backprop

    return Model("with_array_from_list_ragged", forward, init=init)


def _raggeds_to_arrays(X: List[Ragged]) -> Tuple[ArrayXd, List[int]]:
    Xa = [xr.dataXd for xr in X]
    lens = [a.shape[0] for a in Xa]
    return Xa, lens
