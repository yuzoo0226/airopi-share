import dataclasses
import os
import pathlib

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config
from openpi.training import task_sampler as _task_sampler

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


def test_train_skips_val_split_for_fake_data(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],  # noqa: SLF001
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test_skip_fake_split",
        overwrite=True,
        resume=False,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
    )

    def _fail_split(*args, **kwargs):
        raise AssertionError("fake data should not trigger episode splitting")

    monkeypatch.setattr(train, "_split_episodes_by_task", _fail_split)

    train.main(config)


def test_next_batch_fails_fast_on_task_sampler_invariant() -> None:
    class _BrokenIter:
        def __iter__(self):
            return self

        def __next__(self):
            raise _task_sampler.TaskSamplerInvariantError("queue underflow")

    error_state = {"count": 0, "last_error": None}
    with pytest.raises(_task_sampler.TaskSamplerInvariantError):
        train._next_batch(  # noqa: SLF001
            iter(_BrokenIter()),
            data_loader=_BrokenIter(),
            allow_errors=True,
            max_errors=100,
            error_state=error_state,
        )
    assert error_state["count"] == 0


def test_task_feedback_shape_mismatch_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="Per-task feedback shape mismatch"):
        train._assert_task_feedback_compatible(  # noqa: SLF001
            task_indices=np.array([0, 1, 2], dtype=np.int64),
            losses=np.array([0.1, 0.2], dtype=np.float32),
            rank=0,
        )


class _TinyDataset:
    def __init__(self, task_indices: list[int]):
        self._task_indices = list(task_indices)

    def __len__(self) -> int:
        return len(self._task_indices)

    def __getitem__(self, idx: int):
        return {"task_index": self._task_indices[idx]}


def test_sync_adaptive_task_sampler_state_single_process_keeps_local_values() -> None:
    sampler = _task_sampler.AdaptiveTaskSampler(_TinyDataset([0, 1, 0, 1]), seed=0, num_samples=4)
    sampler.set_loss_ema(np.array([2.0, 4.0], dtype=np.float64))

    class _Loader:
        def __init__(self, task_sampler):
            self._task_sampler = task_sampler

    updated = train._sync_adaptive_task_sampler_state(_Loader(sampler))  # noqa: SLF001
    assert updated is True
    np.testing.assert_allclose(sampler.get_loss_ema(), np.array([2.0, 4.0], dtype=np.float64))


def test_sync_adaptive_task_sampler_state_returns_false_without_sampler() -> None:
    class _Loader:
        _task_sampler = None

    updated = train._sync_adaptive_task_sampler_state(_Loader())  # noqa: SLF001
    assert updated is False


def test_concat_feedback_chunks_handles_empty() -> None:
    out = train._concat_feedback_chunks([], dtype=np.int64)  # noqa: SLF001
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_accumulate_task_loss_stats_sums_examples() -> None:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    train._accumulate_task_loss_stats(  # noqa: SLF001
        np.asarray([0, 0, 1], dtype=np.int64),
        np.asarray([1.0, 3.0, 2.0], dtype=np.float64),
        sums,
        counts,
    )
    assert sums == {0: 4.0, 1: 2.0}
    assert counts == {0: 2, 1: 1}


def test_split_episodes_by_task_uses_fast_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    class _FakeFastMeta:
        def __init__(self, repo_id: str, root=None):
            assert repo_id == "dummy/repo"
            self.tasks = {0: "task_a", 1: "task_b"}
            self.episodes = {
                0: {"task_index": 0},
                1: {"task_index": 0},
                2: {"task_index": 1},
                3: {"task_index": 1},
            }

    class _ForbiddenMeta:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LeRobotDatasetMetadata should not be used in split path")

    monkeypatch.setattr(train, "FastLeRobotDatasetMetadata", _FakeFastMeta)
    monkeypatch.setattr(train, "LeRobotDatasetMetadata", _ForbiddenMeta)
    monkeypatch.setenv("OPENPI_EPISODE_SPLIT_CACHE_DIR", str(tmp_path / "split_cache"))

    train_eps, val_eps = train._split_episodes_by_task(
        "dummy/repo",
        val_fraction=0.5,
        seed=0,
        dataset_root=str(tmp_path / "dataset_root"),
    )
    assert set(train_eps + val_eps) == {0, 1, 2, 3}
    assert len(train_eps) == 2
    assert len(val_eps) == 2
