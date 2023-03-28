from thinc.api import Ragged, get_current_ops
import torch

from curated_transformers.models.scalar_weight import build_scalar_weight_v1


def test_scalar_weight_model():
    ops = get_current_ops()
    model = build_scalar_weight_v1(num_layers=2, dropout_prob=0.0)

    with torch.no_grad():
        model.shims[0]._model.layer_weights[0] = 1
        model.shims[0]._model.layer_weights[1] = 1
        model.shims[0]._model.layer_weights[2] = 1

    zeros = ops.alloc2f(15, 2)
    ones = ops.alloc2f(15, 2) + 1

    lens = ops.asarray1i([1, 2, 3, 4, 5])
    X = [
        [
            Ragged(ones.copy(), lens.copy()),
            Ragged(zeros.copy(), lens.copy()),
            Ragged(ones.copy(), lens.copy()),
        ]
    ]
    Y = ops.alloc2f(15, 2) + (1.0 / 3.0) * 2

    Yh, backprop = model(X, is_train=True)
    assert len(Yh) == 1
    ops.xp.testing.assert_array_equal(
        Yh[0].dataXd,
        Y,
    )
    ops.xp.testing.assert_array_equal(
        Yh[0].lengths,
        lens,
    )

    dX = backprop(
        [Ragged(ones.copy(), lengths=lens.copy())],
    )
    dX_expected = ops.alloc2f(15, 2) + (1.0 / 3.0)

    assert len(dX) == 1
    assert len(dX[0]) == 3
    for dX_layer in dX[0]:
        ops.xp.testing.assert_array_equal(
            dX_layer.dataXd,
            dX_expected,
        )
        ops.xp.testing.assert_array_equal(
            dX_layer.lengths,
            lens,
        )
