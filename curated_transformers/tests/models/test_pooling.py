from thinc.api import Ragged, reduce_sum

from curated_transformers.models.pooling import with_ragged_last_layer


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

    ops.xp.testing.assert_equal(dX[0].dataXd, ops.asarray2f([[1.0], [-1.0], [-1.0]]))
    ops.xp.testing.assert_equal(dX[0].lengths, ops.asarray1i([1, 2]))

    ops.xp.testing.assert_equal(dX[1].dataXd, ops.asarray2f([[-1.0], [-1.0], [1.0]]))
    ops.xp.testing.assert_equal(dX[1].lengths, ops.asarray1i([2, 1]))

    ops.xp.testing.assert_equal(dX[2].dataXd, ops.asarray2f([]).reshape(0, 1))
    ops.xp.testing.assert_equal(dX[2].lengths, ops.asarray1i([]))
