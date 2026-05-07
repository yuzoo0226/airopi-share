import numpy as np

from openpi.training import task_log_utils as _task_log_utils


def test_compute_task_counts_sorted() -> None:
    task_indices = np.array([2, 0, 2, 1, 1, 2])
    task_ids, counts = _task_log_utils.compute_task_counts(task_indices)

    assert task_ids.tolist() == [0, 1, 2]
    assert counts.tolist() == [1, 2, 3]


def test_format_task_distribution_basic() -> None:
    task_indices = np.array([0, 0, 1, 2, 2, 2])
    formatted = _task_log_utils.format_task_distribution(task_indices, precision=2)

    assert formatted.startswith("tasks={")
    assert "2:3(0.50)" in formatted
    assert "0:2(0.33)" in formatted
    assert "1:1(0.17)" in formatted


def test_format_task_distribution_truncation() -> None:
    task_indices = np.array([0, 1, 2, 3, 4, 5])
    formatted = _task_log_utils.format_task_distribution(task_indices, max_tasks=3)

    assert formatted.endswith("...}}")
