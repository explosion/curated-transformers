from typing import List
from curated_transformers.models.output import TransformerModelOutput
import pytest
from thinc.api import Model, NumpyOps, Ragged, with_array, chain
from thinc.types import Floats2d

from curated_transformers.models.with_strided_spans import with_strided_spans


def relu_activation() -> Model[Floats2d, Floats2d]:
    def forward(model: Model, X: Floats2d, is_train: bool):
        Y = model.ops.relu(X)

        def backprop(dY: Floats2d) -> Floats2d:
            return dY * model.ops.backprop_relu(dY, Y)

        return Y, backprop

    return Model("relu_activation", forward)


def _add_range() -> Model[Floats2d, Floats2d]:
    """Add range [0, X.size)."""

    def forward(model: Model, X: Floats2d, is_train: bool):
        adds = model.ops.xp.arange(X.size).reshape(X.shape)
        return X + adds, lambda x: x

    return Model("add_range", forward)


def _mock_transformer() -> Model[List[Floats2d], TransformerModelOutput]:
    def forward(model: Model, X: List[Floats2d], is_train: bool):
        return (
            TransformerModelOutput(outputs=[[x] for x in X], last_layer_only=True),
            lambda x: x,
        )

    return Model("mock_transformer", forward)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_with_strided_spans(batch_size):
    ops = NumpyOps()
    trf = chain(with_array(relu_activation()), _mock_transformer())
    model = with_strided_spans(trf, stride=4, window=4, batch_size=batch_size)

    zeros = ops.alloc2f(15, 5)
    ones = ops.alloc2f(15, 5) + 1
    fives = ops.alloc2f(15, 5) + 5

    lengths1 = ops.asarray1i([1, 2, 3, 4, 5])
    lengths2 = ops.asarray1i([5, 4, 3, 2, 1])

    X = [
        Ragged(fives.copy(), lengths=lengths1),
        Ragged(-fives, lengths=lengths2),
        Ragged(fives.copy(), lengths=lengths1.copy()),
        Ragged(-fives, lengths=lengths2.copy()),
    ]
    model.initialize(X)

    Y, backprop = model(X, is_train=True)
    Y = Y.all_outputs
    assert len(Y) == len(X)

    dX = backprop(
        [
            Ragged(ones.copy(), lengths=lengths1),
            Ragged(ones.copy(), lengths=lengths2),
            Ragged(ones.copy(), lengths=lengths1.copy()),
            Ragged(ones.copy(), lengths=lengths2.copy()),
        ]
    )

    for i in [0, 2]:
        ops.xp.testing.assert_array_equal(Y[i][0].data, fives)
        ops.xp.testing.assert_array_equal(Y[i][0].lengths, lengths1)
        ops.xp.testing.assert_array_equal(dX[i].data, ones)
        ops.xp.testing.assert_array_equal(dX[i].lengths, lengths1)
    for i in [1, 3]:
        ops.xp.testing.assert_array_equal(Y[i][0].data, zeros)
        ops.xp.testing.assert_array_equal(Y[i][0].lengths, lengths2)
        ops.xp.testing.assert_array_equal(dX[i].data, zeros)
        ops.xp.testing.assert_array_equal(dX[i].lengths, lengths2)


def test_with_strided_spans_averaging():
    ops = NumpyOps()
    stateful = chain(with_array(_add_range()), _mock_transformer())
    model = with_strided_spans(stateful, stride=2, window=4)

    data = ops.xp.zeros((6, 2))
    lengths = ops.asarray1i([3, 3])
    X = [Ragged(data, lengths=lengths)]

    model.initialize(X)

    Y, backprop = model(X, is_train=False)

    ops.xp.testing.assert_equal(
        Y.all_outputs[0][0].dataXd,
        [[0.0, 1.0], [2.0, 3.0], [6.0, 7.0], [8.0, 9.0], [14.0, 15.0], [16.0, 17.0]],
    )

    ones = data + 1
    dX = backprop(
        [
            Ragged(ones.copy(), lengths=lengths),
        ]
    )
    ops.xp.testing.assert_array_equal(
        dX[0].dataXd,
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.25, 0.25],
            [0.25, 0.25],
            [0.25, 0.25],
            [0.25, 0.25],
        ],
    )
    ops.xp.testing.assert_array_equal(dX[0].lengths, lengths)


def test_incorrect_strides_are_rejected():
    relu = with_array(relu_activation())
    with pytest.raises(ValueError):
        with_strided_spans(relu, stride=2, window=6)
    with pytest.raises(ValueError):
        with_strided_spans(relu, stride=4, window=3)


def test_batch_sizes_are_rejected():
    relu = with_array(relu_activation())
    with pytest.raises(ValueError):
        with_strided_spans(relu, batch_size=-1)
    with pytest.raises(ValueError):
        with_strided_spans(relu, batch_size=0)
