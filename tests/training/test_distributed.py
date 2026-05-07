import os

import pytest

from openpi.training import distributed as _distributed


def _set_required_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    local_rank: int,
    local_world_size: int,
    rank: int,
    world_size: int,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world_size))
    monkeypatch.setenv("MASTER_ADDR", "hnode003")
    monkeypatch.setenv("MASTER_PORT", "31010")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


def test_uuid_cuda_visible_devices_single_local_process_uses_all_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch, local_rank=0, local_world_size=1, rank=0, world_size=2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    captured: dict[str, object] = {}
    monkeypatch.setattr(_distributed.jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(_distributed.jax.distributed, "initialize", lambda **kwargs: captured.update(kwargs))

    ctx = _distributed.initialize_jax_distributed_from_env()

    assert ctx.initialized
    assert captured["local_device_ids"] == [0, 1, 2]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b,GPU-c"


def test_uuid_cuda_visible_devices_multi_local_process_isolates_local_rank_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch, local_rank=1, local_world_size=2, rank=1, world_size=2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    captured: dict[str, object] = {}
    monkeypatch.setattr(_distributed.jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(_distributed.jax.distributed, "initialize", lambda **kwargs: captured.update(kwargs))

    ctx = _distributed.initialize_jax_distributed_from_env()

    assert ctx.initialized
    assert captured["local_device_ids"] == [0]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-b"
