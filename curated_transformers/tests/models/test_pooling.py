from thinc.api import Ragged, reduce_sum

from curated_transformers.models.pooling import (
    with_ragged_last_layer,
    with_ragged_layers,
)


def test_with_ragged_last_layer():
    pooler = with_ragged_last_layer(reduce_sum())
    ops = pooler.ops

    docs = [
        Ragged(ops.asarray1f([1.0, 2.0, 3.0]), lengths=ops.asarray1i([1, 2])),
        Ragged(ops.asarray1f([4.0, 5.0, 6.0]), lengths=ops.asarray1i([2, 1])),
        Ragged(ops.asarray1f([]), lengths=ops.asarray1i([])),
    ]

    Y, backprop = pooler(docs, is_train=True)

    ops.xp.testing.assert_equal(
        Y,
        [
            ops.asarray2f([[1.0], [5.0]]),
            ops.asarray2f([[9.0], [6.0]]),
            ops.asarray2f([]).reshape(0, 1),
        ],
    )

    dX = backprop(
        [
            ops.asarray2f([[1.0], [-1.0]]),
            ops.asarray2f([[-1.0], [1.0]]),
            ops.asarray2f([]).reshape(0, 1),
        ],
    )

    assert len(dX) == 3

    ops.xp.testing.assert_equal(dX[0].dataXd, ops.asarray2f([[1.0], [-1.0], [-1.0]]))
    ops.xp.testing.assert_equal(dX[0].lengths, ops.asarray1i([1, 2]))

    ops.xp.testing.assert_equal(dX[1].dataXd, ops.asarray2f([[-1.0], [-1.0], [1.0]]))
    ops.xp.testing.assert_equal(dX[1].lengths, ops.asarray1i([2, 1]))

    ops.xp.testing.assert_equal(dX[2].dataXd, ops.asarray2f([]).reshape(0, 1))
    ops.xp.testing.assert_equal(dX[2].lengths, ops.asarray1i([]))


def test_with_ragged_layers():
    pooler = with_ragged_layers(reduce_sum())
    ops = pooler.ops

    docs = [
        [
            Ragged(ops.asarray1f([1.0, 2.0, 3.0]), lengths=ops.asarray1i([1, 2])),
            Ragged(ops.asarray1f([3.0, 2.0, 1.0]), lengths=ops.asarray1i([1, 2])),
        ],
        [
            Ragged(ops.asarray1f([4.0, 5.0, 6.0]), lengths=ops.asarray1i([2, 1])),
            Ragged(ops.asarray1f([6.0, 5.0, 4.0]), lengths=ops.asarray1i([2, 1])),
        ],
        [
            Ragged(ops.asarray1f([]), lengths=ops.asarray1i([])),
            Ragged(ops.asarray1f([]), lengths=ops.asarray1i([])),
        ],
    ]

    Y, backprop = pooler(docs, is_train=True)

    ops.xp.testing.assert_equal(
        Y,
        [
            [
                ops.asarray2f([[1.0], [5.0]]),
                ops.asarray2f([[3.0], [3.0]]),
            ],
            [
                ops.asarray2f([[9.0], [6.0]]),
                ops.asarray2f([[11.0], [4.0]]),
            ],
            [
                ops.asarray2f([]).reshape(0, 1),
                ops.asarray2f([]).reshape(0, 1),
            ],
        ],
    )

    dX = backprop(
        [
            [
                ops.asarray2f([[1.0], [-1.0]]),
                ops.asarray2f([[2.0], [-2.0]]),
            ],
            [
                ops.asarray2f([[-1.0], [1.0]]),
                ops.asarray2f([[-2.0], [2.0]]),
            ],
            [
                ops.asarray2f([]).reshape(0, 1),
                ops.asarray2f([]).reshape(0, 1),
            ],
        ],
    )

    assert len(dX) == 3

    assert len(dX[0]) == 2
    ops.xp.testing.assert_equal(dX[0][0].dataXd, ops.asarray2f([[1.0], [-1.0], [-1.0]]))
    ops.xp.testing.assert_equal(dX[0][0].lengths, ops.asarray1i([1, 2]))
    ops.xp.testing.assert_equal(dX[0][1].dataXd, ops.asarray2f([[2.0], [-2.0], [-2.0]]))
    ops.xp.testing.assert_equal(dX[0][1].lengths, ops.asarray1i([1, 2]))

    assert len(dX[1]) == 2
    ops.xp.testing.assert_equal(dX[1][0].dataXd, ops.asarray2f([[-1.0], [-1.0], [1.0]]))
    ops.xp.testing.assert_equal(dX[1][1].dataXd, ops.asarray2f([[-2.0], [-2.0], [2.0]]))
    ops.xp.testing.assert_equal(dX[1][0].lengths, ops.asarray1i([2, 1]))
    ops.xp.testing.assert_equal(dX[1][1].lengths, ops.asarray1i([2, 1]))

    assert len(dX[2]) == 2
    ops.xp.testing.assert_equal(dX[2][0].dataXd, ops.asarray2f([]).reshape(0, 1))
    ops.xp.testing.assert_equal(dX[2][1].dataXd, ops.asarray2f([]).reshape(0, 1))
    ops.xp.testing.assert_equal(dX[2][0].lengths, ops.asarray1i([]))
    ops.xp.testing.assert_equal(dX[2][1].lengths, ops.asarray1i([]))
