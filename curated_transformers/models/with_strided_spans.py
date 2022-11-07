from typing import Callable, List, Tuple, Union
from functools import partial
from thinc.model import Model
from thinc.types import Ragged, Floats2d, Ints1d

from .output import TransformerModelOutput

# In case of a single list, each element corresponds to a single document/span.
# For nested lists, each inner list corresponds to a single document/span.
RaggedInOutT = Union[List[Ragged], List[List[Ragged]]]
Floats2dInOutT = Union[List[Floats2d], List[List[Floats2d]]]


def build_with_strided_spans_v1(
    stride: int = 96, window: int = 128
) -> Callable[
    [Model[List[Ints1d], TransformerModelOutput], int, int],
    Model[List[Ragged], TransformerModelOutput],
]:
    return partial(with_strided_spans, stride=stride, window=window)


def with_strided_spans(
    layer: Model[List[Ints1d], TransformerModelOutput],
    *,
    stride: int = 96,
    window: int = 128,
) -> Model[List[Ragged], TransformerModelOutput]:
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


def with_strided_spans_init(model: Model, X: List[Ragged], Y: List[Ragged]):
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
    model: Model[List[Ragged], TransformerModelOutput],
    X: List[Ragged],
    is_train: bool,
) -> Tuple[TransformerModelOutput, Callable[[RaggedInOutT], RaggedInOutT]]:
    stride: int = model.attrs["stride"]
    window: int = model.attrs["window"]

    spans, doc_lens = _ragged_to_strided_arrays(Xlr=X, stride=stride, window=window)
    # Calculate once and use for all layer outputs.
    doc_len_sums = [int(x.sum()) for x in doc_lens]

    Y_layer, backprop_layer = model.layers[0](spans, is_train=is_train)
    if not isinstance(Y_layer, TransformerModelOutput):
        raise ValueError(f"Unsupported input of type '{type(Y_layer)}'")

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
        Y_layer.all_outputs,
        doc_len_sums,
        stride=stride,
        window=window,
        func=_average_arrays,
    )

    model_output = _strided_arrays_to_ragged(
        model, Y_layer.all_outputs, doc_lens, doc_len_sums, stride=stride
    )
    Y_layer.all_outputs = model_output

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

        dXlf = backprop_layer(dY_spans)
        return _strided_arrays_to_ragged(
            model, dXlf, dY_lengths, dY_length_sums, stride=stride
        )

    return Y_layer, backprop


def _apply_to_overlaps(
    Xlf: Floats2dInOutT,
    len_sums: List[int],
    *,
    stride: int,
    window: int,
    func: Callable,
):
    """Average representations of overlapping windows. This function
    modifies the arrays in Xlf in-place."""

    def _apply_to_layer(input: List[Floats2d]):
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
        _apply_to_layers(Xlf)
    else:
        _apply_to_layer(Xlf)


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
        input: Union[Tuple[Ragged], List[Ragged]], output: List[Floats2d]
    ):
        for Xr in input:
            data = Xr.dataXd
            while data.shape[0] != 0:
                output.append(data[:window])
                data = data[stride:]

    def _apply_to_layers(input: List[List[Ragged]]):
        # Transpose input to reconstruct strides across all documents.
        transposed = list(zip(*Xlr))
        spans = [[] for _ in range(len(transposed))]
        lens = [x.lengths for x in transposed[0]]
        for x, y in zip(transposed, spans):
            _apply_to_layer(x, y)

        # Normalize sequences as lists.
        spans = [[y for y in x] for x in zip(*spans)]
        return spans, lens

    if isinstance(Xlr[0], list):
        # Gradients in the backward pass.
        # Shape of spans: (span, layer)
        return _apply_to_layers(Xlr)
    else:
        # Inputs in the forward pass.
        spans = []
        lens = [x.lengths for x in Xlr]
        _apply_to_layer(Xlr, spans)
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
        input: Union[Tuple[Floats2d], List[Floats2d]], output: List[Ragged]
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
        Xlr = [[] for _ in range(len(transposed))]
        for x, y in zip(transposed, Xlr):
            _apply_to_layer(x, y)

        # Normalize sequences as lists.
        Xlr = [[y for y in x] for x in zip(*Xlr)]
        return Xlr

    if isinstance(Xlf[0], list):
        # Transformer output in forward pass.
        return _apply_to_layers(Xlf)
    else:
        # Gradient from torch in backward pass.
        Xlr = []
        _apply_to_layer(Xlf, Xlr)
        return Xlr
