from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, cast
from functools import partial
from thinc.backends import NumpyOps
from thinc.model import Model
from thinc.types import Ragged, Floats2d, Ints1d

from .output import TransformerModelOutput
from .types import (
    RaggedInOutT,
    Floats2dInOutT,
    SpanExtractorBackpropT,
    SpanExtractorInT,
    SpanExtractorOutT,
    SpanExtractorModelT,
    TorchTransformerInT,
    TorchTransformerModelT,
    TorchTransformerOutT,
)


_NUMPY_OPS = NumpyOps()


def build_with_strided_spans_v1(
    stride: int = 96, window: int = 128, batch_size: int = 384
) -> Callable[[TorchTransformerModelT, int, int], SpanExtractorModelT]:
    """Construct a model that can be used to convert a list of ragged
    piece identifiers to strided spans and vice-versa.

    stride (int):
        The stride of the span generator.
    window (int):
        The (maximum) size of each span.
    batch_size (int):
        The maximum number of spans that are processed
        by the transformer model at a time.
    """
    return partial(
        with_strided_spans, stride=stride, window=window, batch_size=batch_size
    )


def with_strided_spans(
    trf_model: TorchTransformerModelT,
    *,
    stride: int = 96,
    window: int = 128,
    batch_size: int = 384,
) -> Model[List[Ragged], TransformerModelOutput]:
    if not (window // 2 <= stride <= window):
        raise ValueError(
            f"Stride must be within [window / 2, window] ([{window // 2}, {window}]), was: {stride}"
        )
    if batch_size <= 0:
        raise ValueError("Span batch size must greater than zero")

    attrs = {
        "stride": stride,
        "window": window,
        "batch_size": batch_size,
    }
    return Model(
        "with_strided_spans",
        with_strided_spans_forward,
        init=with_strided_spans_init,
        layers=[trf_model],
        attrs=attrs,
    )


def with_strided_spans_init(
    model: SpanExtractorModelT,
    X: Optional[SpanExtractorInT],
    Y: Optional[SpanExtractorInT],
):
    stride: int = model.attrs["stride"]
    window: int = model.attrs["window"]

    X_spans = (
        _ragged_to_strided_arrays(X, stride=stride, window=window).spans
        if X is not None
        else None
    )
    Y_spans = (
        _ragged_to_strided_arrays(Y, stride=stride, window=window).spans
        if Y is not None
        else None
    )
    model.layers[0].initialize(X=X_spans, Y=Y_spans)


def with_strided_spans_forward(
    model: SpanExtractorModelT,
    X: SpanExtractorInT,
    is_train: bool,
) -> Tuple[SpanExtractorOutT, SpanExtractorBackpropT]:
    transformer: TorchTransformerModelT = model.layers[0]
    stride: int = model.attrs["stride"]
    window: int = model.attrs["window"]
    batch_size: int = model.attrs["batch_size"]

    spans = _ragged_to_strided_arrays(Xlr=X, stride=stride, window=window)

    backprops = []
    outputs = []
    for batch in _split_spans(spans.spans, batch_size):
        output, bp = transformer(cast(TorchTransformerInT, batch), is_train=is_train)
        if not isinstance(output, TransformerModelOutput):
            raise ValueError(f"Unsupported input of type '{type(output)}'")
        outputs.append(output)
        backprops.append(bp)

    trf_output = _unsplit_outputs(outputs)

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
        trf_output.all_outputs,
        docs_strides=spans.docs_strides,
        docs_windows=spans.docs_windows,
        func=_average_arrays,
    )

    model_output = _strided_arrays_to_ragged(
        model, trf_output.all_outputs, spans.doc_lens, docs_strides=spans.docs_strides
    )
    trf_output.all_outputs = cast(List[List[Ragged]], model_output)

    def backprop(dY: RaggedInOutT):
        dY_spans = _ragged_to_strided_arrays(dY, stride=stride, window=window)

        # Both overlaps will have dY. However, the proper gradient is 0.5 * dY
        # since we averaged the overlaps in the forward pass.
        _apply_to_overlaps(
            dY_spans.spans,
            docs_strides=dY_spans.docs_strides,
            docs_windows=dY_spans.docs_windows,
            func=_normalize_gradients,
        )

        dXlf = []
        split_dY_spans = list(_split_spans(dY_spans.spans, batch_size))
        assert len(split_dY_spans) == len(backprops)
        for dY_batch, bp_trf in zip(split_dY_spans, backprops):
            dXlf_batch = bp_trf(dY_batch)
            dXlf.extend(dXlf_batch)

        return _strided_arrays_to_ragged(
            model, dXlf, dY_spans.doc_lens, docs_strides=dY_spans.docs_strides
        )

    return trf_output, backprop


def _split_spans(spans: Floats2dInOutT, batch_size: int) -> Iterable[Floats2dInOutT]:
    while len(spans):
        batch = spans[:batch_size]
        yield batch
        spans = spans[batch_size:]


def _unsplit_outputs(outputs: List[TorchTransformerOutT]) -> TorchTransformerOutT:
    last_layer_only = outputs[0].last_layer_only
    merged = TransformerModelOutput(
        outputs=cast(
            List[List[Floats2d]],
            [inner for outer in outputs for inner in outer.all_outputs],
        ),
        last_layer_only=last_layer_only,
    )
    return merged


def _apply_to_overlaps(
    Xlf: Floats2dInOutT,
    *,
    docs_strides: List[Ints1d],
    docs_windows: List[Ints1d],
    func: Callable[[Any, Any], None],
):
    """Applies a function to overlapping windows, modifying the arrays
    in Xlf in-place."""

    def _apply_to_layer(input: Union[Tuple[Floats2d, ...], List[Floats2d]]):
        for doc_strides, doc_windows in zip(docs_strides, docs_windows):
            # Suppose that we have two spans:
            #
            # <--- stride1 --->
            # <----- window1 ----->
            #                  <--- stride2 --->
            #                  <----- window2 ----->
            #
            # Then window1_overlap is the part of window1 that overlaps with
            # window2, window2_overlap is the part of window2 that overlaps
            # with window1.
            window1_overlap = None
            for stride, window in zip(doc_strides, doc_windows):
                if window1_overlap is not None:
                    window2_overlap = input[0][: window1_overlap.shape[0]]
                    func(window1_overlap, window2_overlap)

                overlap_len = window - stride
                window1_overlap = input[0][-overlap_len:] if overlap_len else None
                input = input[1:]

    def _apply_to_layers(input: List[List[Floats2d]]):
        # We need to transpose the input since the overlaps happen between
        # each layer of each span, i.e., span 1 layer 1 overlaps with span 2 layer 1, etc.
        transposed = list(zip(*input))
        for x in transposed:
            _apply_to_layer(x)

    if isinstance(Xlf[0], list):
        nested_list = cast(List[List[Floats2d]], Xlf)
        _apply_to_layers(nested_list)
    else:
        flat_tuple = cast(Tuple[Floats2d, ...], Xlf)
        _apply_to_layer(flat_tuple)


def _average_arrays(array1, array2):
    array1 += array2
    array1 /= 2.0
    array2[:] = array1


def _normalize_gradients(array1, array2):
    array1 *= 0.5
    array2 *= 0.5


@dataclass
class _Spans:
    __slots__ = ["doc_lens", "docs_strides", "docs_windows", "spans"]

    doc_lens: List[Ints1d]
    docs_strides: List[Ints1d]
    docs_windows: List[Ints1d]
    spans: Floats2dInOutT


def _ragged_to_strided_arrays(Xlr: RaggedInOutT, *, stride: int, window: int) -> _Spans:
    """Convert the per-Doc Ragged sequences to an array of sub-sequences that span
    all the input documents."""

    def _apply_to_layer(
        input: Union[Tuple[Ragged, ...], List[Ragged]],
        docs_strides: List[Ints1d],
        docs_windows: List[Ints1d],
        output: List[Floats2d],
    ):
        for Xr, doc_strides, doc_windows in zip(input, docs_strides, docs_windows):
            data = cast(Floats2d, Xr.dataXd)
            for doc_stride, doc_window in zip(doc_strides, doc_windows):
                output.append(data[:doc_window])
                data = data[doc_stride:]

    def _apply_to_layers(input: List[List[Ragged]]):
        # Transpose input to reconstruct strides across all documents.
        transposed = list(zip(*Xlr))
        spans: List[List[Floats2d]] = [[] for _ in range(len(transposed))]
        lens = [x.lengths for x in transposed[0]]
        docs_strides, docs_windows = _find_spans_with_token_boundaries(
            transposed[0], stride, window
        )
        for x, y in zip(transposed, spans):
            _apply_to_layer(x, docs_strides, docs_windows, y)

        # Normalize sequences as lists.
        spans = [[y for y in x] for x in zip(*spans)]
        return _Spans(
            doc_lens=lens,
            docs_strides=docs_strides,
            docs_windows=docs_windows,
            spans=spans,
        )

    if isinstance(Xlr[0], list):
        # Gradients in the backward pass.
        # Shape of spans: (span, layer)
        nested_list = cast(List[List[Ragged]], Xlr)
        return _apply_to_layers(nested_list)
    else:
        # Inputs in the forward pass.
        flat_list = cast(List[Ragged], Xlr)
        docs_strides, docs_windows = _find_spans_with_token_boundaries(
            flat_list, stride, window
        )
        spans: List[Floats2d] = []
        lens = [x.lengths for x in flat_list]
        _apply_to_layer(flat_list, docs_strides, docs_windows, spans)
        return _Spans(
            doc_lens=lens,
            docs_strides=docs_strides,
            docs_windows=docs_windows,
            spans=spans,
        )


def _strided_arrays_to_ragged(
    model: Model,
    Xlf: Floats2dInOutT,
    lens: List[Ints1d],
    *,
    docs_strides: List[Ints1d],
) -> RaggedInOutT:
    """Inverse operation of _ragged_to_strided_arrays."""

    def _apply_to_layer(
        input: Union[Tuple[Floats2d, ...], List[Floats2d]], output: List[Ragged]
    ):
        for Xr_lens, doc_strides in zip(lens, docs_strides):
            arrs = []
            for stride in doc_strides:
                arr = input[0][:stride]
                arrs.append(arr)
                input = input[1:]
            output.append(Ragged(model.ops.flatten(arrs), lengths=Xr_lens))

    def _apply_to_layers(input: List[List[Floats2d]]):
        # Transpose input to reconstruct strides across all documents.
        transposed = list(zip(*input))
        Xlr: List[List[Ragged]] = [[] for _ in range(len(transposed))]
        for x, y in zip(transposed, Xlr):
            _apply_to_layer(x, y)

        # Normalize sequences as lists.
        Xlr = [[y for y in x] for x in zip(*Xlr)]
        return Xlr

    if isinstance(Xlf[0], list):
        # Transformer output in forward pass.
        nested_list = cast(List[List[Floats2d]], Xlf)
        return _apply_to_layers(nested_list)
    else:
        # Gradient from torch in backward pass.
        flat_tuple = cast(Tuple[Floats2d, ...], Xlf)
        Xlr: List[Ragged] = []
        _apply_to_layer(flat_tuple, Xlr)
        return Xlr


def _find_spans_with_token_boundaries(
    Xlr: Union[Tuple[Ragged, ...], List[Ragged]], stride: int, window: int
):
    """When splitting inputs on a stride/window, we may be splitting up a
    token consisting of multiple pieces. This function computes the
    strides/windows rounded up to a token boundary."""
    xp = _NUMPY_OPS.xp
    docs_strides = []
    docs_windows = []
    for Xr in Xlr:
        doc_strides = []
        doc_windows = []

        # We find the next boundary by first computing the cumulative
        # sums of the token lengths. We can then (binary) search the token
        # that lies on the stride/window boundary. Suppose that we have
        # a stride of 128, there are three scenarios:
        #
        # 1. A particular token has a cumulative length of 128. In this case,
        #    splitting on the stride wouldn't split up a token.
        # 2. No token has a cumulative length of 128, but there is a token
        #    with a cumulative length >128. The binary search will return the
        #    index of the first token token with length >128, say 130. We
        #    will then use a stride of 130 to include the full token.
        # 3. There is no token with a cumulative length >=128. Binary search
        #    will return the length of the array. In this case, we use the
        #    remaining sequence.

        cumsums = _NUMPY_OPS.asarray1i(xp.cumsum(Xr.lengths))
        while cumsums.size:
            indices = xp.searchsorted(cumsums, xp.array([stride, window]))
            # Ensure that we don't index out of bounds in scenario (3).
            vals = cumsums[xp.minimum(indices, cumsums.size - 1)]

            doc_strides.append(vals[0])
            doc_windows.append(vals[1])

            # Re-use the cumulative sums for the next iteration.
            cumsums = cumsums[indices[0] + 1 :]
            cumsums -= vals[0]

        docs_strides.append(_NUMPY_OPS.asarray1i(doc_strides))
        docs_windows.append(_NUMPY_OPS.asarray1i(doc_windows))

    return docs_strides, docs_windows
