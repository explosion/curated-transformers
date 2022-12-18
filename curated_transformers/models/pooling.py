from typing import List, Union
import numpy
from spacy.tokens import Doc
from thinc.api import Model
from thinc.types import Floats2d, Ragged


from .types import AllOutputsPoolingModelT, PoolingModelT


def pool_all_outputs(
    pooling: PoolingModelT,
) -> AllOutputsPoolingModelT:
    return Model("pool_all_outputs", _pool_all_outputs_forward, layers=[pooling])


def _pool_all_outputs_forward(
    model: AllOutputsPoolingModelT, X: List[Doc], is_train: bool
):
    pooling: PoolingModelT = model.layers[0]
    xp = model.ops.xp

    datas = []
    lens = []
    doc_layers = []
    doc_lens = []
    for X_doc in X:
        X_doc_layers = X_doc._.trf_data.all_outputs if isinstance(X_doc, Doc) else X_doc
        for X_layer in X_doc_layers:
            datas.append(X_layer.dataXd)
            lens.append(X_layer.lengths)
        doc_layers.append(len(X_doc_layers))
        doc_lens.append(len(X_doc_layers[0].lengths))

    X_flat = Ragged(xp.concatenate(datas, axis=0), xp.concatenate(lens, axis=0))
    Y_pooled, pooling_backprop = pooling(X_flat, is_train)

    doc_layer_lens = numpy.asarray(
        [doc_layer * doc_len for doc_layer, doc_len in zip(doc_layers, doc_lens)]
    )
    Y = [
        xp.split(doc_array, doc_layer)
        for (doc_array, doc_layer) in zip(
            xp.split(Y_pooled, numpy.cumsum(doc_layer_lens)[:-1]),
            doc_layers,
        )
    ]

    def backprop(dY):
        dY_pooled_flat = xp.concatenate(
            [dY_layer for dY_doc in dY for dY_layer in dY_doc]
        )
        dY_flat = pooling_backprop(dY_pooled_flat).dataXd

        dY = []
        for X_doc in X:
            dY_layer = []

            X_doc_layers = (
                X_doc._.trf_data.all_outputs if isinstance(X_doc, Doc) else X_doc
            )
            for X_layer in X_doc_layers:
                doc_unpooled_len = X_layer.dataXd.shape[0]
                dY_layer.append(Ragged(dY_flat[:doc_unpooled_len], X_layer.lengths))
                dY_flat = dY_flat[doc_unpooled_len:]

            dY.append(dY_layer)

        return dY

    return Y, backprop
