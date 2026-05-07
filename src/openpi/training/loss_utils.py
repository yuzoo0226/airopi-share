from __future__ import annotations

import jax.numpy as jnp

from openpi.shared import array_typing as at


def reduce_per_example_loss(loss: at.Array) -> at.Array:
    """Reduce a loss tensor to per-example values.

    Args:
        loss: Loss tensor with shape (batch, ...) or (batch,).

    Returns:
        Per-example loss with shape (batch,).
    """
    if loss.ndim <= 1:
        return loss
    axes = tuple(range(1, loss.ndim))
    return jnp.mean(loss, axis=axes)
