from thinc.api import get_current_ops

from curated_transformers.models.distill import MSELoss


def test_mse_loss_with_averaging():
    ops = get_current_ops()
    y = ops.asarray([[-1.0, 0.0, 1.0, 1.0]])
    y_h = ops.asarray([[-0.5, -0.5, 0.0, 1.0]])
    loss = MSELoss(ops, normalization="mean")

    loss_value, grads = loss([y_h], [y])
    ops.xp.testing.assert_almost_equal(loss_value, 0.375)
    ops.xp.testing.assert_equal(ops.asarray(grads), [[[0.125, -0.125, -0.25, 0.0]]])


def test_mse_loss_with_squared_l2_norm():
    ops = get_current_ops()
    y = ops.asarray([[-1.0, 0.0, 1.0, 1.0]])
    y_h = ops.asarray([[-0.5, -0.5, 0.0, 1.0]])
    loss = MSELoss(ops, normalization="squared_l2_norm")

    loss_value, grads = loss([y_h], [y])
    ops.xp.testing.assert_almost_equal(loss_value, 0.5)
    assert ops.xp.allclose(
        ops.asarray(grads),
        [[[0.16666, -0.16666, -0.33333, 0.0]]],
        atol=1e-5,
        rtol=1e-5,
    )
