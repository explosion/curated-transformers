from thinc.api import Model, NumpyOps, Ragged, with_array
from thinc.types import Floats2d

from spacy_experimental.transformers.models.with_strided_spans import with_strided_spans


def relu_activation() -> Model[Floats2d, Floats2d]:
    def forward(model: Model, X: Floats2d, is_train: bool):
        Y = model.ops.relu(X)

        def backprop(dY: Floats2d) -> Floats2d:
            return dY * model.ops.backprop_relu(dY, Y)

        return Y, backprop

    return Model("relu_activation", forward)


def test_with_strided_spans():
    ops = NumpyOps()
    relu = with_array(relu_activation())
    model = with_strided_spans(relu, stride=2, window=4)

    zeros = ops.alloc2f(15, 5)
    ones = ops.alloc2f(15, 5) + 1
    fives = ops.alloc2f(15, 5) + 5

    lengths1 = ops.asarray1i([1, 2, 3, 4, 5])
    lengths2 = ops.asarray1i([5, 4, 3, 2, 1])

    X = [
        Ragged(fives, lengths=lengths1),
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
            Ragged(ones, lengths=lengths1),
            Ragged(ones, lengths=lengths2),
        ]
    )
    ops.xp.testing.assert_array_equal(dX[0].data, ones)
    ops.xp.testing.assert_array_equal(dX[1].data, zeros)
    ops.xp.testing.assert_array_equal(dX[0].lengths, lengths1)
    ops.xp.testing.assert_array_equal(dX[1].lengths, lengths2)
