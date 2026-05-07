import numpy as np
import torch

from openpi.training import task_sampler as _task_sampler


class SimpleTaskDataset:
    def __init__(self, task_indices: np.ndarray) -> None:
        self._task_indices = task_indices

    def __len__(self) -> int:
        return len(self._task_indices)

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"task_index": int(self._task_indices[index])}


class EpisodeMetadataTaskDataset:
    def __init__(self) -> None:
        self.meta = type(
            "Meta",
            (),
            {
                "episodes": {
                    10: {"tasks": ["task_a"]},
                    11: {"tasks": ["task_b"]},
                },
                "task_to_task_index": {"task_a": 0, "task_b": 1},
                "tasks": {0: "task_a", 1: "task_b"},
            },
        )()
        self.episodes = [10, 11]
        self.episode_data_index = {
            "from": torch.LongTensor([0, 2]),
            "to": torch.LongTensor([2, 5]),
        }

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict[str, int]:
        raise AssertionError("Metadata path should avoid frame-level __getitem__ scan.")


def test_uniform_task_sampler_pop_task_indices() -> None:
    """Uniform sampler should emit task indices for queued samples."""
    tasks = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    dataset = SimpleTaskDataset(tasks)
    sampler = _task_sampler.UniformTaskSampler(dataset, seed=0, num_samples=4)

    _ = list(iter(sampler))
    task_ids = sampler.pop_task_indices(4)

    assert task_ids.shape == (4,)
    assert set(np.asarray(task_ids)).issubset(set(tasks.tolist()))


def test_adaptive_task_sampler_updates_probs() -> None:
    """Adaptive sampler should increase probability for higher-loss tasks."""
    tasks = np.array([0, 1, 0, 1])
    dataset = SimpleTaskDataset(tasks)
    sampler = _task_sampler.AdaptiveTaskSampler(dataset, seed=0, num_samples=4, alpha=1.0)

    sampler.update(np.array([0, 1, 1, 1]), np.array([1.0, 10.0, 10.0, 10.0]))

    assert sampler._task_probs[1] > sampler._task_probs[0]


def test_uniform_task_sampler_distribution() -> None:
    """Uniform sampler should be approximately uniform across tasks."""
    tasks = np.array([0, 1, 2, 3] * 8)
    dataset = SimpleTaskDataset(tasks)
    num_samples = 4000
    sampler = _task_sampler.UniformTaskSampler(dataset, seed=1, num_samples=num_samples)

    _ = list(iter(sampler))
    sampled_tasks = sampler.pop_task_indices(num_samples)
    counts = np.bincount(sampled_tasks, minlength=4)
    proportions = counts / num_samples

    assert np.all(np.abs(proportions - 0.25) < 0.05)


def test_adaptive_task_sampler_prioritizes_high_loss() -> None:
    """Adaptive sampler should favor tasks with higher loss."""
    tasks = np.array([0, 1] * 16)
    dataset = SimpleTaskDataset(tasks)
    num_samples = 2000
    sampler = _task_sampler.AdaptiveTaskSampler(dataset, seed=2, num_samples=num_samples, alpha=1.0, ema_decay=0.0)

    sampler.update(
        np.array([0, 1, 1, 1, 1, 1]),
        np.array([1.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
    )

    _ = list(iter(sampler))
    sampled_tasks = sampler.pop_task_indices(num_samples)
    counts = np.bincount(sampled_tasks, minlength=2)
    proportions = counts / num_samples

    assert proportions[1] > 0.65


def test_extract_task_index_from_episode_metadata() -> None:
    dataset = EpisodeMetadataTaskDataset()
    task_indices = _task_sampler._extract_task_index(dataset)
    assert np.array_equal(task_indices, np.array([0, 0, 1, 1, 1], dtype=np.int64))


def test_uniform_task_sampler_global_sync_disjoint_per_step() -> None:
    tasks = np.arange(80, dtype=np.int64) % 4
    dataset = SimpleTaskDataset(tasks)

    sampler_rank0 = _task_sampler.UniformTaskSampler(
        dataset,
        seed=123,
        num_samples=8,
        num_replicas=2,
        rank=0,
        global_batch_size=8,
        local_batch_size=4,
        sampling_mode="global_sync",
        disjoint_per_step=True,
    )
    sampler_rank1 = _task_sampler.UniformTaskSampler(
        dataset,
        seed=123,
        num_samples=8,
        num_replicas=2,
        rank=1,
        global_batch_size=8,
        local_batch_size=4,
        sampling_mode="global_sync",
        disjoint_per_step=True,
    )

    indices_rank0 = list(iter(sampler_rank0))
    indices_rank1 = list(iter(sampler_rank1))

    assert len(indices_rank0) == 8
    assert len(indices_rank1) == 8

    for step in range(2):
        start = step * 4
        end = start + 4
        assert set(indices_rank0[start:end]).isdisjoint(set(indices_rank1[start:end]))

    task_ids_rank0 = sampler_rank0.pop_task_indices(8)
    task_ids_rank1 = sampler_rank1.pop_task_indices(8)
    assert task_ids_rank0.shape == (8,)
    assert task_ids_rank1.shape == (8,)


def test_task_sampler_underflow_raises_invariant_error() -> None:
    tasks = np.array([0, 1, 0, 1])
    dataset = SimpleTaskDataset(tasks)
    sampler = _task_sampler.UniformTaskSampler(dataset, seed=0, num_samples=2)

    _ = list(iter(sampler))
    with np.testing.assert_raises(_task_sampler.TaskSamplerInvariantError):
        sampler.pop_task_indices(3)
