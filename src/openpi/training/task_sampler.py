from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import logging
from typing import Any

import numpy as np
import torch


class TaskSamplerInvariantError(RuntimeError):
    """Raised when task-sampler invariants are violated and recovery is unsafe."""


def _unwrap_dataset(dataset: Any) -> Any:
    seen: set[int] = set()
    while hasattr(dataset, "_dataset") and id(dataset) not in seen:
        seen.add(id(dataset))
        dataset = getattr(dataset, "_dataset")
    return dataset


def _try_get_column(dataset: Any, key: str) -> np.ndarray | None:
    try:
        column = dataset[key]
    except Exception:
        return None
    try:
        return np.asarray(column)
    except Exception:
        return None


def _resolve_episode_task_index(meta: Any, episode: Any) -> int | None:
    if not isinstance(episode, dict):
        return None
    task_index = episode.get("task_index")
    if task_index is not None:
        return int(task_index)

    task_label: str | None = None
    tasks = episode.get("tasks")
    if isinstance(tasks, list) and tasks:
        task_label = str(tasks[0])
    elif isinstance(tasks, str) and tasks:
        task_label = tasks
    elif episode.get("task"):
        task_label = str(episode["task"])
    if not task_label:
        return None

    task_to_index = getattr(meta, "task_to_task_index", None)
    if isinstance(task_to_index, dict):
        idx = task_to_index.get(task_label)
        if idx is not None:
            return int(idx)

    tasks_map = getattr(meta, "tasks", None)
    if isinstance(tasks_map, dict):
        for idx, label in tasks_map.items():
            if str(label) == task_label:
                return int(idx)
    return None


def _extract_task_index_from_episode_metadata(base: Any) -> np.ndarray | None:
    """Build frame-level task indices from episode metadata when available.

    This avoids scanning every frame via __getitem__, which is very expensive for
    parquet-backed datasets on multi-node runs.
    """
    meta = getattr(base, "meta", None)
    episode_data_index = getattr(base, "episode_data_index", None)
    episodes_meta = getattr(meta, "episodes", None) if meta is not None else None
    if meta is None or episode_data_index is None or episodes_meta is None:
        return None
    if not isinstance(episode_data_index, dict):
        return None

    starts_raw = episode_data_index.get("from")
    ends_raw = episode_data_index.get("to")
    if starts_raw is None or ends_raw is None:
        return None

    starts = np.asarray(starts_raw, dtype=np.int64)
    ends = np.asarray(ends_raw, dtype=np.int64)
    if starts.ndim != 1 or ends.ndim != 1 or starts.size != ends.size:
        return None
    if starts.size == 0:
        return np.empty(0, dtype=np.int64)
    if starts[0] != 0 or np.any(ends < starts):
        return None
    if starts.size > 1 and np.any(starts[1:] != ends[:-1]):
        return None

    total = int(ends[-1])
    if total <= 0:
        return np.empty(0, dtype=np.int64)

    episodes = getattr(base, "episodes", None)
    if episodes is not None and len(episodes) != starts.size:
        return None

    task_indices = np.empty(total, dtype=np.int64)
    for ep_data_idx, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        if end <= start:
            continue
        ep_idx = int(episodes[ep_data_idx]) if episodes is not None else ep_data_idx
        episode = episodes_meta.get(ep_idx) if hasattr(episodes_meta, "get") else None
        if episode is None:
            return None
        task_index = _resolve_episode_task_index(meta, episode)
        if task_index is None:
            return None
        task_indices[start:end] = task_index

    return task_indices


def _extract_task_index(dataset: Any) -> np.ndarray:
    base = _unwrap_dataset(dataset)

    for attr in ("hf_dataset", "_hf_dataset", "dataset", "_dataset", "data"):
        obj = getattr(base, attr, None)
        if obj is None:
            continue
        column = _try_get_column(obj, "task_index")
        if column is not None:
            return column.astype(np.int64)

    task_indices = _extract_task_index_from_episode_metadata(base)
    if task_indices is not None:
        logging.info(
            "Built task index mapping from episode metadata (%d samples).",
            len(task_indices),
        )
        return task_indices

    if not hasattr(base, "__len__") or not hasattr(base, "__getitem__"):
        raise ValueError("Dataset does not support random access; cannot build task index mapping.")

    task_indices = np.empty(len(base), dtype=np.int64)
    for idx in range(len(base)):
        item = base[idx]
        if "task_index" not in item:
            raise ValueError('Task sampler requires "task_index" in dataset items.')
        task_indices[idx] = int(item["task_index"])
    return task_indices


def _resolve_sampling_mode(mode: str, *, num_replicas: int) -> str:
    if mode == "auto":
        return "global_sync" if num_replicas > 1 else "per_rank"
    if mode in ("global_sync", "per_rank"):
        return mode
    raise ValueError(f"Unknown task sampler multi-process mode: {mode}")


class TaskSamplerBase(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset: Any,
        *,
        seed: int,
        num_samples: int | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        global_batch_size: int | None = None,
        local_batch_size: int | None = None,
        sampling_mode: str = "per_rank",
        disjoint_per_step: bool = True,
    ) -> None:
        super().__init__()
        self._seed = int(seed)
        self._epoch = 0
        self._index_queue: deque[int] = deque()

        task_index = _extract_task_index(dataset)
        self._task_index_by_dataset_index = task_index
        self._task_ids = np.unique(task_index)
        self._task_ids.sort()
        self._task_id_to_pos = {int(tid): pos for pos, tid in enumerate(self._task_ids.tolist())}
        self._indices_by_task = [np.where(task_index == tid)[0] for tid in self._task_ids]

        if num_samples is None:
            num_samples = len(task_index)
        self._num_samples = int(num_samples)

        self._num_replicas = int(num_replicas)
        self._rank = int(rank)
        if self._num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {self._num_replicas}.")
        if self._rank < 0 or self._rank >= self._num_replicas:
            raise ValueError(f"rank must be in [0, {self._num_replicas}), got {self._rank}.")

        self._sampling_mode = _resolve_sampling_mode(sampling_mode, num_replicas=self._num_replicas)
        self._disjoint_per_step = bool(disjoint_per_step)

        if self._sampling_mode == "global_sync":
            if self._num_replicas <= 1:
                raise ValueError("global_sync mode requires num_replicas > 1.")
            if global_batch_size is None or local_batch_size is None:
                raise ValueError(
                    "global_sync mode requires both global_batch_size and local_batch_size."
                )
            self._global_batch_size = int(global_batch_size)
            self._local_batch_size = int(local_batch_size)
            if self._global_batch_size != self._local_batch_size * self._num_replicas:
                raise ValueError(
                    "global_batch_size must equal local_batch_size * num_replicas, got "
                    f"global_batch_size={self._global_batch_size}, "
                    f"local_batch_size={self._local_batch_size}, "
                    f"num_replicas={self._num_replicas}."
                )
            if self._local_batch_size <= 0:
                raise ValueError(f"local_batch_size must be positive, got {self._local_batch_size}.")
        else:
            self._global_batch_size = (
                int(global_batch_size)
                if global_batch_size is not None
                else int(local_batch_size) if local_batch_size is not None else 1
            )
            self._local_batch_size = (
                int(local_batch_size)
                if local_batch_size is not None
                else int(global_batch_size) if global_batch_size is not None else 1
            )

        self._logged_disjoint_fallback = False

        logging.info(
            "Initialized task sampler with %d tasks across %d samples. "
            "world_size=%d, global_batch=%d, local_batch=%d, mode=%s, disjoint=%s",
            len(self._task_ids),
            len(task_index),
            self._num_replicas,
            self._global_batch_size,
            self._local_batch_size,
            self._sampling_mode,
            self._disjoint_per_step,
        )

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def reset_queue(self) -> None:
        self._index_queue.clear()

    def pop_task_indices(self, batch_size: int) -> np.ndarray:
        if len(self._index_queue) < batch_size:
            raise TaskSamplerInvariantError(
                "Task sampler index queue underflow: "
                f"need {batch_size}, have {len(self._index_queue)}, "
                f"rank={self._rank}, world_size={self._num_replicas}."
            )
        indices = [self._index_queue.popleft() for _ in range(batch_size)]
        return self._task_index_by_dataset_index[np.asarray(indices, dtype=np.int64)]

    def _record_index(self, index: int) -> None:
        self._index_queue.append(int(index))

    def _make_rng(self) -> np.random.Generator:
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        return rng

    def _sample_task_positions(self, rng: np.random.Generator, num_draws: int) -> np.ndarray:
        return rng.integers(0, len(self._task_ids), size=num_draws, dtype=np.int64)

    def _sample_indices_for_task_positions(
        self,
        rng: np.random.Generator,
        task_positions: np.ndarray,
        *,
        disjoint: bool,
    ) -> np.ndarray:
        task_positions = np.asarray(task_positions, dtype=np.int64).reshape(-1)
        out = np.empty(task_positions.shape[0], dtype=np.int64)

        for task_pos in np.unique(task_positions):
            task_pos_i = int(task_pos)
            mask = np.where(task_positions == task_pos_i)[0]
            count = int(mask.size)
            if count == 0:
                continue

            candidates = self._indices_by_task[task_pos_i]
            if disjoint and count <= len(candidates):
                selected = rng.choice(candidates, size=count, replace=False)
            elif disjoint and count > len(candidates):
                if not self._logged_disjoint_fallback:
                    logging.warning(
                        "Disjoint-per-step sampling fallback on task_id=%d: requested=%d, available=%d. "
                        "Sampling with replacement for overflow.",
                        int(self._task_ids[task_pos_i]),
                        count,
                        len(candidates),
                    )
                    self._logged_disjoint_fallback = True
                unique = rng.choice(candidates, size=len(candidates), replace=False)
                overflow = rng.choice(candidates, size=count - len(candidates), replace=True)
                selected = np.concatenate([unique, overflow])
                rng.shuffle(selected)
            else:
                selected = rng.choice(candidates, size=count, replace=True)

            out[mask] = np.asarray(selected, dtype=np.int64)

        return out

    def _iter_per_rank(self, rng: np.random.Generator) -> Iterable[int]:
        task_positions = self._sample_task_positions(rng, self._num_samples)
        indices = self._sample_indices_for_task_positions(rng, task_positions, disjoint=False)
        for index in indices.tolist():
            self._record_index(index)
            yield index

    def _iter_global_sync(self, rng: np.random.Generator) -> Iterable[int]:
        emitted = 0
        rank_start = self._rank * self._local_batch_size
        rank_end = rank_start + self._local_batch_size

        while emitted < self._num_samples:
            task_positions = self._sample_task_positions(rng, self._global_batch_size)
            global_indices = self._sample_indices_for_task_positions(
                rng,
                task_positions,
                disjoint=self._disjoint_per_step,
            )
            local_indices = global_indices[rank_start:rank_end]

            for index in local_indices.tolist():
                if emitted >= self._num_samples:
                    break
                self._record_index(index)
                emitted += 1
                yield index

    def __iter__(self) -> Iterable[int]:
        rng = self._make_rng()
        if self._sampling_mode == "global_sync":
            yield from self._iter_global_sync(rng)
            return
        yield from self._iter_per_rank(rng)


class UniformTaskSampler(TaskSamplerBase):
    pass


class AdaptiveTaskSampler(TaskSamplerBase):
    def __init__(
        self,
        dataset: Any,
        *,
        seed: int,
        num_samples: int | None = None,
        alpha: float = 1.0,
        ema_decay: float = 0.9,
        min_prob: float = 0.0,
        eps: float = 1e-6,
        num_replicas: int = 1,
        rank: int = 0,
        global_batch_size: int | None = None,
        local_batch_size: int | None = None,
        sampling_mode: str = "per_rank",
        disjoint_per_step: bool = True,
    ) -> None:
        super().__init__(
            dataset,
            seed=seed,
            num_samples=num_samples,
            num_replicas=num_replicas,
            rank=rank,
            global_batch_size=global_batch_size,
            local_batch_size=local_batch_size,
            sampling_mode=sampling_mode,
            disjoint_per_step=disjoint_per_step,
        )
        self._alpha = float(alpha)
        self._ema_decay = float(ema_decay)
        self._min_prob = float(min_prob)
        self._eps = float(eps)

        self._loss_ema = np.ones(len(self._task_ids), dtype=np.float64)
        self._task_probs = np.full(len(self._task_ids), 1.0 / len(self._task_ids), dtype=np.float64)

    def _sample_task_positions(self, rng: np.random.Generator, num_draws: int) -> np.ndarray:
        return rng.choice(len(self._task_ids), size=num_draws, p=self._task_probs)

    def update(self, task_indices: np.ndarray, losses: np.ndarray) -> None:
        task_indices = np.asarray(task_indices, dtype=np.int64).reshape(-1)
        losses = np.asarray(losses, dtype=np.float64).reshape(-1)

        if task_indices.shape[0] != losses.shape[0]:
            raise ValueError(
                f"Task sampler update expects same length for task_indices and losses, got "
                f"{task_indices.shape[0]} and {losses.shape[0]}."
            )

        for task_id in np.unique(task_indices):
            pos = self._task_id_to_pos.get(int(task_id))
            if pos is None:
                continue
            mask = task_indices == task_id
            mean_loss = losses[mask].mean()
            self._loss_ema[pos] = self._ema_decay * self._loss_ema[pos] + (1.0 - self._ema_decay) * mean_loss

        self._recompute_probs()

    def _recompute_probs(self) -> None:
        weights = np.power(self._loss_ema + self._eps, self._alpha)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            self._task_probs = np.full_like(weights, 1.0 / len(weights))
            return

        probs = weights / weights_sum
        if self._min_prob > 0:
            total_min = self._min_prob * len(probs)
            if total_min >= 1.0:
                probs = np.full_like(probs, 1.0 / len(probs))
            else:
                probs = probs * (1.0 - total_min) + self._min_prob

        probs /= probs.sum()
        self._task_probs = probs

    def get_loss_ema(self) -> np.ndarray:
        return self._loss_ema.copy()

    def set_loss_ema(self, loss_ema: np.ndarray) -> None:
        loss_ema = np.asarray(loss_ema, dtype=np.float64).reshape(-1)
        if loss_ema.shape != self._loss_ema.shape:
            raise ValueError(
                f"AdaptiveTaskSampler.set_loss_ema expects shape {self._loss_ema.shape}, got {loss_ema.shape}."
            )
        self._loss_ema = loss_ema.copy()
        self._recompute_probs()


def create_task_sampler(
    config: Any,
    dataset: Any,
    *,
    seed: int,
    num_replicas: int = 1,
    rank: int = 0,
    global_batch_size: int | None = None,
    local_batch_size: int | None = None,
) -> TaskSamplerBase:
    kind = getattr(config, "kind", "none")
    mode = _resolve_sampling_mode(getattr(config, "multi_process_mode", "auto"), num_replicas=num_replicas)
    disjoint_per_step = bool(getattr(config, "disjoint_per_step", True))

    common_kwargs = dict(
        seed=seed,
        num_replicas=num_replicas,
        rank=rank,
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        sampling_mode=mode,
        disjoint_per_step=disjoint_per_step,
    )

    if kind == "uniform":
        return UniformTaskSampler(dataset, **common_kwargs)
    if kind == "adaptive":
        return AdaptiveTaskSampler(
            dataset,
            alpha=getattr(config, "alpha", 1.0),
            ema_decay=getattr(config, "ema_decay", 0.9),
            min_prob=getattr(config, "min_prob", 0.0),
            eps=getattr(config, "eps", 1e-6),
            **common_kwargs,
        )
    raise ValueError(f"Unknown task sampler kind: {kind}")
