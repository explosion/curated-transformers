from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, cast
from functools import partial
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
from ..errors import Errors


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
            Errors.E017.format(
                stride=stride, half_window_size=window // 2, window_size=window
            )
        )
    if batch_size <= 0:
        raise ValueError(Errors.E018)

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
    model: SpanExtractorModelT,
    X: SpanExtractorInT,
    is_train: bool,
) -> Tuple[SpanExtractorOutT, SpanExtractorBackpropT]:
    transformer: TorchTransformerModelT = model.layers[0]
    stride: int = model.attrs["stride"]
    window: int = model.attrs["window"]
    batch_size: int = model.attrs["batch_size"]

    spans, doc_lens = _ragged_to_strided_arrays(Xlr=X, stride=stride, window=window)
    # Calculate once and use for all layer outputs.
    doc_len_sums = [int(x.sum()) for x in doc_lens]

    backprops = []
    outputs = []
    for batch in _split_spans(spans, batch_size):
        output, bp = transformer(cast(TorchTransformerInT, batch), is_train=is_train)
        if not isinstance(output, TransformerModelOutput):
            raise ValueError(
                Errors.E014.format(model_name=model.name, input_type=type(output))
            )
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
        doc_len_sums,
        stride=stride,
        window=window,
        func=_average_arrays,
    )

    model_output = _strided_arrays_to_ragged(
        model, trf_output.all_outputs, doc_lens, doc_len_sums, stride=stride
    )
    trf_output.all_outputs = cast(List[List[Ragged]], model_output)

    def backprop(dY: RaggedInOutT):
        dY_spans, dY_lengths = _ragged_to_strided_arrays(
            dY, stride=stride, window=window
        )
        # Calculate once and use for all layer outputs.
        dY_length_sums = [int(x.sum()) for x in dY_lengths]

        # Both overlaps will have dY. However, the proper gradient is 0.5 * dY
        # since we averaged the overlaps in the forward pass.
        _apply_to_overlaps(
            dY_spans,
            dY_length_sums,
            stride=stride,
            window=window,
            func=_normalize_gradients,
        )

        dXlf = []
        split_dY_spans = list(_split_spans(dY_spans, batch_size))
        assert len(split_dY_spans) == len(backprops)
        for dY_batch, bp_trf in zip(split_dY_spans, backprops):
            dXlf_batch = bp_trf(dY_batch)
            dXlf.extend(dXlf_batch)

        return _strided_arrays_to_ragged(
            model, dXlf, dY_lengths, dY_length_sums, stride=stride
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
    len_sums: List[int],
    *,
    stride: int,
    window: int,
    func: Callable[[Any, Any], None],
):
    """Applies a function to overlapping windows, modifying the arrays
    in Xlf in-place."""

    def _apply_to_layer(input: Union[Tuple[Floats2d, ...], List[Floats2d]]):
        for doc_len in len_sums:
            prev_overlap = None
            while doc_len != 0:
                overlap = min(window - stride, doc_len)
                if prev_overlap is not None:
                    cur_overlap = input[0][:overlap]
                    prev_overlap = prev_overlap[-overlap:]
                    func(prev_overlap, cur_overlap)
                prev_overlap = input[0][-overlap:]

                doc_len -= min(stride, doc_len)
                input = input[1:]

    def _apply_to_layers(input: List[List[Floats2d]]):
        # We need to transpose the input since the overlaps happen between
        # each layer of each span, i.e., span 1 layer 1 overlaps with span 2 layer 1, etc.
        transposed = list(zip(*input))
        for x in transposed:
            _apply_to_layer(x)

    # Nothing to do if there is no overlap.
    if window - stride == 0:
        return

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


def _ragged_to_strided_arrays(
    Xlr: RaggedInOutT, *, stride: int, window: int
) -> Tuple[Floats2dInOutT, List[Ints1d]]:
    """Convert the per-Doc Ragged sequences to an array of sub-sequences that span
    all the input documents."""

    def _apply_to_layer(
        input: Union[Tuple[Ragged, ...], List[Ragged]], output: List[Floats2d]
    ):
        for Xr in input:
            data = cast(Floats2d, Xr.dataXd)
            while data.shape[0] != 0:
                output.append(data[:window])
                data = data[stride:]

    def _apply_to_layers(input: List[List[Ragged]]):
        # Transpose input to reconstruct strides across all documents.
        transposed = list(zip(*Xlr))
        spans: List[List[Floats2d]] = [[] for _ in range(len(transposed))]
        lens = [x.lengths for x in transposed[0]]
        for x, y in zip(transposed, spans):
            _apply_to_layer(x, y)

        # Normalize sequences as lists.
        spans = [[y for y in x] for x in zip(*spans)]
        return spans, lens

    if isinstance(Xlr[0], list):
        # Gradients in the backward pass.
        # Shape of spans: (span, layer)
        nested_list = cast(List[List[Ragged]], Xlr)
        return _apply_to_layers(nested_list)
    else:
        # Inputs in the forward pass.
        flat_list = cast(List[Ragged], Xlr)
        spans: List[Floats2d] = []
        lens = [x.lengths for x in flat_list]
        _apply_to_layer(flat_list, spans)
        return spans, lens


def _strided_arrays_to_ragged(
    model: Model,
    Xlf: Floats2dInOutT,
    lens: List[Ints1d],
    len_sums: List[int],
    *,
    stride: int,
) -> RaggedInOutT:
    """Inverse operation of _ragged_to_strided_arrays."""

    def _apply_to_layer(
        input: Union[Tuple[Floats2d, ...], List[Floats2d]], output: List[Ragged]
    ):
        for doc_len, Xr_lens in zip(len_sums, lens):
            arrs = []
            while doc_len != 0:
                arr = input[0][: min(stride, doc_len)]
                arrs.append(arr)
                doc_len -= arr.shape[0]
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
