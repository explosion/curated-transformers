from typing import List
import numpy
from spacy.tokens import Doc
from thinc.api import Model
from thinc.types import Floats2d, Ragged


def pool_all_outputs(
    pooling: Model[Ragged, Floats2d],
) -> Model[List[Doc], List[List[Floats2d]]]:
    return Model("pool_all_outputs", _pool_all_outputs_forward, layers=[pooling])


def _pool_all_outputs_forward(
    model: Model[List[Doc], List[List[Floats2d]]], X: List[Doc], is_train: bool
):
    pooling: Model[Ragged, Floats2d] = model.layers[0]
    xp = model.ops.xp

    datas = [X_layer.dataXd for X_doc in X for X_layer in X_doc._.trf_data.all_outputs]
    lens = [X_layer.lengths for X_doc in X for X_layer in X_doc._.trf_data.all_outputs]
    doc_layers = [X_doc._.trf_data.num_outputs for X_doc in X]
    doc_lens = [len(X_doc._.trf_data.last_hidden_layer_state.lengths) for X_doc in X]

    X_flat = Ragged(
        xp.concatenate(datas, axis=0), xp.concatenate(lens, axis=0)
    )
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
        dY_pooled_flat = xp.concatenate([dY_layer for dY_doc in dY for dY_layer in dY_doc])
        dY_flat = pooling_backprop(dY_pooled_flat)
        # We now have one large Ragged. Loop over the inputs and consume this
        # Ragged with the lenghts of the inputs.

    return Y, None
