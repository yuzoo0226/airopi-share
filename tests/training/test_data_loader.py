import dataclasses
from typing import Any

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


class TaskIndexDataset:
    def __init__(self, base: Any, task_indices: np.ndarray) -> None:
        self._base = base
        self._task_indices = task_indices

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self._base[index])
        item["task_index"] = int(self._task_indices[index])
        return item


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_infer_fast_lerobot_include_columns_from_repack_transform() -> None:
    data_config = _config.DataConfig(
        action_sequence_keys=("action.relative",),
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "head_rgb": "observation.image.head",
                        "hand_rgb": "observation.image.hand",
                        "state": "observation.state",
                        "actions": "action.relative",
                    }
                )
            ]
        ),
    )
    columns = _data_loader._infer_fast_lerobot_include_columns(data_config)  # noqa: SLF001
    assert columns is not None
    assert "observation.image.head" in columns
    assert "observation.image.hand" in columns
    assert "observation.state" in columns
    assert "action.relative" in columns
    assert "timestamp" in columns
    assert "task_index" in columns


def test_infer_fast_lerobot_include_columns_without_repack_returns_none() -> None:
    data_config = _config.DataConfig(action_sequence_keys=("action.relative",))
    assert _data_loader._infer_fast_lerobot_include_columns(data_config) is None  # noqa: SLF001


def test_uniform_task_sampler_emits_task_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config.get_config("debug")
    config = dataclasses.replace(
        config,
        batch_size=4,
        num_workers=0,
        task_sampler=_config.TaskSamplerConfig(kind="uniform"),
    )

    base_dataset = _data_loader.FakeDataset(config.model, 16)
    task_indices = np.arange(len(base_dataset)) % 4
    dataset = TaskIndexDataset(base_dataset, task_indices)

    def fake_create_torch_dataset(data_config, action_horizon, model_config, *, episodes=None):
        return dataset

    monkeypatch.setattr(_data_loader, "create_torch_dataset", fake_create_torch_dataset)

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2, shuffle=True)
    batches = list(loader)

    assert len(batches) == 2

    for observation, actions, task_ids in batches:
        assert observation.state.shape[0] == config.batch_size
        assert actions.shape[0] == config.batch_size
        assert task_ids.shape[0] == config.batch_size
        assert set(np.asarray(task_ids)).issubset(set(range(4)))


def test_uniform_task_sampler_multiple_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config.get_config("debug")
    config = dataclasses.replace(
        config,
        batch_size=4,
        num_workers=0,
        task_sampler=_config.TaskSamplerConfig(kind="uniform"),
    )

    base_dataset = _data_loader.FakeDataset(config.model, 32)
    task_indices = np.arange(len(base_dataset)) % 4
    dataset = TaskIndexDataset(base_dataset, task_indices)

    def fake_create_torch_dataset(data_config, action_horizon, model_config, *, episodes=None):
        return dataset

    monkeypatch.setattr(_data_loader, "create_torch_dataset", fake_create_torch_dataset)

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=4, shuffle=True)
    data_iter = iter(loader)

    for _ in range(4):
        _ = next(data_iter)


def test_none_task_sampler_emits_task_indices_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config.get_config("debug")
    config = dataclasses.replace(
        config,
        batch_size=4,
        num_workers=0,
        task_sampler=_config.TaskSamplerConfig(kind="none"),
    )

    base_dataset = _data_loader.FakeDataset(config.model, 16)
    task_indices = np.arange(len(base_dataset)) % 4
    dataset = TaskIndexDataset(base_dataset, task_indices)

    def fake_create_torch_dataset(data_config, action_horizon, model_config, *, episodes=None):
        return dataset

    monkeypatch.setattr(_data_loader, "create_torch_dataset", fake_create_torch_dataset)

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=1, shuffle=True)
    batch = next(iter(loader))
    assert isinstance(batch, tuple) and len(batch) == 3
    observation, actions, task_ids = batch
    assert observation.state.shape[0] == config.batch_size
    assert actions.shape[0] == config.batch_size
    assert task_ids.shape[0] == config.batch_size


class _FakeShard:
    def __init__(self, batch_size: int) -> None:
        self.data = np.zeros((batch_size, 16, 32), dtype=np.float32)


class _FakeGlobalActions:
    def __init__(self) -> None:
        self.shape = (64, 16, 32)
        self.is_fully_addressable = False
        self.addressable_shards = [_FakeShard(4), _FakeShard(4)]


class _SingleBatchLoader:
    def __init__(self, actions: _FakeGlobalActions) -> None:
        self._actions = actions

    def __iter__(self):
        yield {
            "image": {
                "base_0_rgb": np.zeros((64, 1, 1, 3), dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": np.ones((64,), dtype=bool),
            },
            "state": np.zeros((64, 32), dtype=np.float32),
            "actions": self._actions,
        }


class _RecordingTaskSampler:
    def __init__(self) -> None:
        self.pop_sizes: list[int] = []

    def pop_task_indices(self, batch_size: int) -> np.ndarray:
        self.pop_sizes.append(batch_size)
        return np.arange(batch_size, dtype=np.int64)


def test_data_loader_impl_uses_process_local_batch_for_task_indices() -> None:
    actions = _FakeGlobalActions()
    recording_sampler = _RecordingTaskSampler()
    loader = _data_loader.DataLoaderImpl(
        _config.DataConfig(repo_id="fake"),
        _SingleBatchLoader(actions),
        task_sampler=recording_sampler,
        include_task_index=True,
    )

    observation, returned_actions, task_indices = next(iter(loader))
    assert observation.state.shape[0] == 64
    assert returned_actions is actions
    assert task_indices.shape == (8,)
    assert recording_sampler.pop_sizes == [8]
