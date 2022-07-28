from typing import List
from functools import partial
from thinc.model import Model
from thinc.types import Ragged


def build_with_strided_spans(stride=96, window=128):
    return partial(with_strided_spans, stride=stride, window=window)


def with_strided_spans(
    layer, *, stride=96, window=128
) -> Model[List[Ragged], List[Ragged]]:
    attrs = {
        "stride": stride,
        "window": window,
    }
    return Model(
        "with_strided_spans",
        with_strided_spans_forward,
        init=with_strided_spans_init,
        layers=[layer],
        attrs=attrs,
    )


def with_strided_spans_init(model: Model, X, Y):
    stride: int = model.attrs["stride"]
    window: int = model.attrs["window"]

    spans = []
    for doc in X:
        data = doc.dataXd
        while data.shape[0] != 0:
            spans.append(data[:window])
            data = data[stride:]

    if X:
        model.layers[0].initialize(X=spans)


def with_strided_spans_forward(
    model: Model[List[Ragged], List[Ragged]], X: List[Ragged], is_train: bool
):
    stride: int = model.attrs["stride"]
    window: int = model.attrs["window"]

    spans = []
    doc_lens = []
    for doc in X:
        doc_len = 0
        data = doc.dataXd
        while data.shape[0] != 0:
            doc_len += 1
            spans.append(data[:window])
            data = data[stride:]
        doc_lens.append(doc_len)

    Y_layer, backprop_layer = model.layers[0](spans, is_train=is_train)

    def backprop(dY):
        raise RuntimeError("backprop is not yet implemented")

    # Todo: mean of previous window and current stride (overlapping part)?
    Y_docs = []
    for doc, doc_len in zip(X, doc_lens):
        Y_doc = [y[:stride] for y in Y_layer[:doc_len]]
        Y_docs.append(Ragged(model.ops.flatten(Y_doc), lengths=doc.lengths))
        Y_layer = Y_layer[doc_len:]

    return Y_docs, backprop
