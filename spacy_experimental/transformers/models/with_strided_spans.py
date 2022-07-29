from typing import List, Tuple
from functools import partial
from thinc.model import Model
from thinc.types import Ragged, Floats2d, Ints1d


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

    X_spans = (
        _ragged_to_strided_arrays(X, stride=stride, window=window)[0]
        if X is not None
        else None
    )
    Y_spans = (
        _ragged_to_strided_arrays(Y, stride=stride, window=window)[0]
        if Y is not None
        else None
    )
    model.layers[0].initialize(X=X_spans, Y=Y_spans)


def with_strided_spans_forward(
    model: Model[List[Ragged], List[Ragged]], X: List[Ragged], is_train: bool
):
    stride: int = model.attrs["stride"]
    window: int = model.attrs["window"]

    spans, doc_lens = _ragged_to_strided_arrays(Xlr=X, stride=stride, window=window)

    Y_layer, backprop_layer = model.layers[0](spans, is_train=is_train)

    def backprop(dY):
        dY_spans, dY_lengths = _ragged_to_strided_arrays(
            dY, stride=stride, window=window
        )
        dXlf = backprop_layer(dY_spans)
        return _strided_arrays_to_ragged(
            model, dXlf, dY_lengths, stride=stride, window=window
        )

    Y_docs = _strided_arrays_to_ragged(
        model, Y_layer, doc_lens, stride=stride, window=window
    )

    return Y_docs, backprop


def _ragged_to_strided_arrays(
    Xlr: List[Ragged], *, stride: int, window: int
) -> Tuple[List[Floats2d], List[int]]:
    spans = []
    lens = []
    doc_lens = []
    for Xr in Xlr:
        doc_len = 0
        data = Xr.dataXd
        lens.append(Xr.lengths)
        while data.shape[0] != 0:
            doc_len += 1
            spans.append(data[:window])
            data = data[stride:]
        doc_lens.append(doc_len)

    return spans, lens


def _strided_arrays_to_ragged(
    model: Model, Xlf: List[Floats2d], lens: List[Ints1d], *, stride: int, window: int
) -> List[Ragged]:
    # Todo: mean of previous window and current stride (overlapping part)?
    Xlr = []
    for Xr_lens in lens:
        doc_len = int(Xr_lens.sum())
        arrs = []
        while doc_len != 0:
            arr = Xlf[0][: min(stride, doc_len)]
            arrs.append(arr)
            doc_len -= arr.shape[0]
            Xlf = Xlf[1:]
        Xlr.append(Ragged(model.ops.flatten(arrs), lengths=Xr_lens))

    return Xlr
