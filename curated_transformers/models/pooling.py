from typing import List, Union
import numpy
from spacy.tokens import Doc
from thinc.api import Model
from thinc.types import Ragged


from .types import WithRaggedLayersModelT, WithRaggedLastLayerModelT, PoolingModelT


def with_ragged_layers(
    pooling: PoolingModelT,
) -> WithRaggedLayersModelT:
    """Concatenate all Docs and layers into a Ragged for pooling."""
    return Model("with_ragged_layers", with_ragged_layers_forward, layers=[pooling])


def with_ragged_layers_forward(
    model: WithRaggedLayersModelT,
    X: Union[List[Doc], List[List[Ragged]]],
    is_train: bool,
):
    pooling: PoolingModelT = model.layers[0]
    xp = model.ops.xp

    datas = []
    lens = []
    doc_layer_counts = []
    doc_token_counts = []
    for X_doc in X:
        X_doc_layers = X_doc._.trf_data.all_outputs if isinstance(X_doc, Doc) else X_doc
        for X_layer in X_doc_layers:
            datas.append(X_layer.dataXd)
            lens.append(X_layer.lengths)
        doc_layer_counts.append(len(X_doc_layers))
        doc_token_counts.append(len(X_doc_layers[0].lengths))

    X_flat = Ragged(xp.concatenate(datas, axis=0), xp.concatenate(lens, axis=0))
    Y_pooled, pooling_backprop = pooling(X_flat, is_train)

    # This array is placed in CPU memory intentionally --- CuPy split (used
    # below) does not support indices in device memory.
    doc_layer_lens = numpy.asarray(
        [
            doc_layer * doc_len
            for doc_layer, doc_len in zip(doc_layer_counts, doc_token_counts)
        ]
    )
    Y = [
        xp.split(doc_array, doc_layer)
        for (doc_array, doc_layer) in zip(
            xp.split(Y_pooled, numpy.cumsum(doc_layer_lens)[:-1]),
            doc_layer_counts,
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


def with_ragged_last_layer(
    pooling: PoolingModelT,
) -> WithRaggedLastLayerModelT:
    """Concatenate the last layers for all Docs into a Ragged for pooling."""
    return Model(
        "with_ragged_last_layer", with_ragged_last_layer_forward, layers=[pooling]
    )


def with_ragged_last_layer_forward(
    model: WithRaggedLastLayerModelT, X: Union[List[Doc], List[Ragged]], is_train: bool
):
    pooling: PoolingModelT = model.layers[0]
    xp = model.ops.xp

    datas = []
    lens = []
    doc_lens = []
    for X_doc in X:
        X_doc_layer = (
            X_doc._.trf_data.last_hidden_layer_state
            if isinstance(X_doc, Doc)
            else X_doc
        )
        datas.append(X_doc_layer.dataXd)
        lens.append(X_doc_layer.lengths)
        doc_lens.append(len(X_doc_layer.lengths))

    X_flat = Ragged(xp.concatenate(datas, axis=0), xp.concatenate(lens, axis=0))
    Y_pooled, pooling_backprop = pooling(X_flat, is_train)
    Y = xp.split(Y_pooled, numpy.cumsum(doc_lens)[:-1])

    def backprop(dY):
        dY_pooled_flat = xp.concatenate(dY)
        dY_flat = pooling_backprop(dY_pooled_flat).dataXd

        dY = []
        for X_doc in X:
            X_doc_layer = (
                X_doc._.trf_data.last_hidden_layer_states
                if isinstance(X_doc, Doc)
                else X_doc
            )
            doc_unpooled_len = X_doc_layer.dataXd.shape[0]
            dY.append(Ragged(dY_flat[:doc_unpooled_len], X_doc_layer.lengths))
            dY_flat = dY_flat[doc_unpooled_len:]

        return dY

    return Y, backprop
