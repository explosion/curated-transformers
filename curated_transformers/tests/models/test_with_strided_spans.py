import pytest
from thinc.api import Model, NumpyOps, Ragged, with_array
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


def test_with_strided_spans():
    ops = NumpyOps()
    relu = with_array(relu_activation())
    model = with_strided_spans(relu, stride=4, window=4)

    zeros = ops.alloc2f(15, 5)
    ones = ops.alloc2f(15, 5) + 1
    fives = ops.alloc2f(15, 5) + 5

    lengths1 = ops.asarray1i([1, 2, 3, 4, 5])
    lengths2 = ops.asarray1i([5, 4, 3, 2, 1])

    X = [
        Ragged(fives.copy(), lengths=lengths1),
        Ragged(-fives, lengths=lengths2),
    ]
    model.initialize(X)

    Y, backprop = model(X, is_train=True)
    ops.xp.testing.assert_array_equal(Y[0].data, fives)
    ops.xp.testing.assert_array_equal(Y[1].data, zeros)
    ops.xp.testing.assert_array_equal(Y[0].lengths, lengths1)
    ops.xp.testing.assert_array_equal(Y[1].lengths, lengths2)

    dX = backprop(
        [
            Ragged(ones.copy(), lengths=lengths1),
            Ragged(ones.copy(), lengths=lengths2),
        ]
    )
    ops.xp.testing.assert_array_equal(dX[0].data, ones)
    ops.xp.testing.assert_array_equal(dX[1].data, zeros)
    ops.xp.testing.assert_array_equal(dX[0].lengths, lengths1)
    ops.xp.testing.assert_array_equal(dX[1].lengths, lengths2)


def test_with_strided_spans_averaging():
    ops = NumpyOps()
    stateful = with_array(_add_range())
    model = with_strided_spans(stateful, stride=2, window=4)

    data = ops.xp.zeros((6, 2))
    lengths = ops.asarray1i([3, 3])
    X = [Ragged(data, lengths=lengths)]

    model.initialize(X)

    Y, backprop = model(X, is_train=False)

    ops.xp.testing.assert_equal(
        Y[0].dataXd,
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
