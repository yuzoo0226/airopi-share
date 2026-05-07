import numpy as np
import jax.numpy as jnp

from openpi.training import loss_utils as _loss_utils


def test_reduce_per_example_loss_1d() -> None:
    loss = jnp.asarray([1.0, 2.0, 3.0])
    reduced = _loss_utils.reduce_per_example_loss(loss)
    np.testing.assert_allclose(np.asarray(reduced), np.asarray(loss))


def test_reduce_per_example_loss_2d() -> None:
    loss = jnp.asarray([[1.0, 2.0], [3.0, 5.0]])
    reduced = _loss_utils.reduce_per_example_loss(loss)
    expected = np.asarray([1.5, 4.0])
    np.testing.assert_allclose(np.asarray(reduced), expected)


def test_reduce_per_example_loss_3d() -> None:
    loss = jnp.asarray([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
    reduced = _loss_utils.reduce_per_example_loss(loss)
    expected = np.asarray([4.0, 5.0])
    np.testing.assert_allclose(np.asarray(reduced), expected)
