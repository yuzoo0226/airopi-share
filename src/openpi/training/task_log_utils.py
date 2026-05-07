from __future__ import annotations

import numpy as np


def compute_task_counts(task_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute task ids and counts from a batch of task indices.

    Args:
        task_indices: Array of task indices with shape (batch,).

    Returns:
        A tuple of (task_ids, counts), both sorted by task id.
    """
    task_indices = np.asarray(task_indices, dtype=np.int64).reshape(-1)
    task_ids, counts = np.unique(task_indices, return_counts=True)
    return task_ids, counts


def format_task_distribution(
    task_indices: np.ndarray,
    *,
    max_tasks: int = 8,
    precision: int = 2,
) -> str:
    """Format task distribution for logging.

    Args:
        task_indices: Array of task indices with shape (batch,).
        max_tasks: Maximum number of tasks to include in the output.
        precision: Decimal precision for proportions.

    Returns:
        A compact string containing counts and proportions.
    """
    task_indices = np.asarray(task_indices, dtype=np.int64).reshape(-1)
    if task_indices.size == 0:
        return "tasks=none"

    task_ids, counts = np.unique(task_indices, return_counts=True)
    total = counts.sum()

    order = np.lexsort((task_ids, -counts))
    task_ids = task_ids[order]
    counts = counts[order]

    shown = min(len(task_ids), max_tasks)
    parts = []
    for task_id, count in zip(task_ids[:shown], counts[:shown], strict=True):
        proportion = count / total
        parts.append(f"{int(task_id)}:{int(count)}({proportion:.{precision}f})")

    suffix = ", ..." if len(task_ids) > shown else ""
    return f"tasks={{" + ", ".join(parts) + suffix + "}}"
