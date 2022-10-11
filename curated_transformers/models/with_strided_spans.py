from typing import Callable, List, Tuple
from functools import partial
from thinc.model import Model
from thinc.types import Ragged, Floats2d, Ints1d


def build_with_strided_spans_v1(stride=96, window=128):
    return partial(with_strided_spans, stride=stride, window=window)


def with_strided_spans(
    layer, *, stride=96, window=128
) -> Model[List[Ragged], List[Ragged]]:
    if not (window // 2 <= stride <= window):
        raise ValueError(
            f"Stride must be within [window / 2, window] ([{window // 2}, {window}]), was: {stride}"
        )

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

    # Suppose we have:
    #
    # A B C D E F G H I
    # <--->   stride
    # <-----> window
    #
    # D is part of both the current and the next window. To
    # ensure that there is no drift in representations, we
    # average them in both representations.
    _apply_to_overlaps(
        Y_layer, doc_lens, stride=stride, window=window, func=_average_arrays
    )

    Y_docs = _strided_arrays_to_ragged(
        model, Y_layer, doc_lens, stride=stride, window=window
    )

    def backprop(dY):
        dY_spans, dY_lengths = _ragged_to_strided_arrays(
            dY, stride=stride, window=window
        )

        # Both overlaps will have dY. However, the proper gradient is 0.5 * dY
        # since we averaged the overlaps in the forward pass.
        _apply_to_overlaps(
            dY_spans,
            dY_lengths,
            stride=stride,
            window=window,
            func=_normalize_gradients,
        )

        dXlf = backprop_layer(dY_spans)
        return _strided_arrays_to_ragged(
            model, dXlf, dY_lengths, stride=stride, window=window
        )

    return Y_docs, backprop


def _apply_to_overlaps(
    Xlf: List[Floats2d],
    lens: List[Ints1d],
    *,
    stride: int,
    window: int,
    func: Callable,
):
    """Average representations of overlapping windows. This function
    modifies the arrays in Xlf in-place."""
    if window - stride == 0:
        # Nothing to do if there is no overlap.
        return

    for Xr_lens in lens:
        doc_len = int(Xr_lens.sum())
        prev_overlap = None
        while doc_len != 0:
            overlap = min(window - stride, doc_len)
            if prev_overlap is not None:
                cur_overlap = Xlf[0][:overlap]
                prev_overlap = prev_overlap[-overlap:]
                func(prev_overlap, cur_overlap)
            prev_overlap = Xlf[0][-overlap:]

            doc_len -= min(stride, doc_len)
            Xlf = Xlf[1:]


def _average_arrays(array1, array2):
    array1 += array2
    array1 /= 2.0
    array2[:] = array1


def _normalize_gradients(array1, array2):
    array1 *= 0.5
    array2 *= 0.5


def _ragged_to_strided_arrays(
    Xlr: List[Ragged], *, stride: int, window: int
) -> Tuple[List[Floats2d], List[List[int]]]:
    spans = []
    lens = []
    for Xr in Xlr:
        data = Xr.dataXd
        lens.append(Xr.lengths)
        while data.shape[0] != 0:
            spans.append(data[:window])
            data = data[stride:]

    return spans, lens


def _strided_arrays_to_ragged(
    model: Model, Xlf: List[Floats2d], lens: List[Ints1d], *, stride: int, window: int
) -> List[Ragged]:
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
