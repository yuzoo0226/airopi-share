import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from pyarrow import parquet as pq

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.video_utils import encode_video_frames

from openpi.training.fast_lerobot_dataset import FastLeRobotDataset
from openpi.training.fast_lerobot_metadata import FastLeRobotDatasetMetadata


def _coerce_video_support(tmp_path):
    try:
        import av  # noqa: F401
        from PIL import Image
    except ImportError:
        return False

    img_dir = tmp_path / "probe_frames"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(img_dir / "frame_000000.png")
    video_path = tmp_path / "probe.mp4"
    try:
        encode_video_frames(img_dir, video_path, fps=1, overwrite=True)
    except Exception:  # noqa: BLE001
        return False
    return video_path.is_file()


def _build_tiny_dataset(tmp_path, *, repo_id: str, use_videos: bool) -> tuple[str, str, int]:
    root = tmp_path / repo_id
    fps = 10
    image_dtype = "video" if use_videos else "image"
    features = {
        "actions": {"dtype": "float32", "shape": (2,), "names": ["a0", "a1"]},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["s0", "s1", "s2"]},
        "observation.images.cam0": {
            "dtype": image_dtype,
            "shape": (4, 5, 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        use_videos=use_videos,
    )

    for ep_idx in range(2):
        for frame_idx in range(3):
            frame = {
                "actions": np.array([ep_idx, frame_idx], dtype=np.float32),
                "observation.state": np.array([ep_idx, frame_idx, ep_idx + frame_idx], dtype=np.float32),
                "observation.images.cam0": np.full(
                    (4, 5, 3),
                    fill_value=(ep_idx * 10 + frame_idx),
                    dtype=np.uint8,
                ),
                "task": f"task-{ep_idx}",
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    return repo_id, str(root), fps


def _compare_items(standard_item: dict, fast_item: dict) -> None:
    assert set(standard_item) == set(fast_item)
    for key in standard_item:
        standard_value = standard_item[key]
        fast_value = fast_item[key]
        if isinstance(standard_value, torch.Tensor):
            torch.testing.assert_close(fast_value, standard_value)
        else:
            assert fast_value == standard_value


@pytest.mark.parametrize("backend", ["parquet", "hf"])
def test_fast_lerobot_matches_standard_images(tmp_path, backend, monkeypatch):
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_OOB", "error")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1")
    repo_id, root, fps = _build_tiny_dataset(
        tmp_path,
        repo_id=f"test_fast_lerobot_image_{backend}",
        use_videos=False,
    )
    delta_timestamps = {
        "actions": [0.0, 1.0 / fps],
        "observation.state": [0.0, 2.0 / fps],
        "observation.images.cam0": [0.0],
    }

    standard = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    fast = FastLeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
        lerobot_backend=backend,
        lerobot_checks="all",
    )

    assert len(standard) == len(fast)
    for idx in range(len(standard)):
        _compare_items(standard[idx], fast[idx])

    with pytest.raises(IndexError):
        _ = standard[len(standard)]
    with pytest.raises(IndexError):
        _ = fast[len(fast)]


def test_fast_lerobot_matches_standard_videos(tmp_path, monkeypatch):
    if not _coerce_video_support(tmp_path):
        pytest.skip("PyAV encoder not available for video tests.")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_OOB", "error")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1")

    repo_id, root, fps = _build_tiny_dataset(
        tmp_path,
        repo_id="test_fast_lerobot_video",
        use_videos=True,
    )
    delta_timestamps = {
        "actions": [0.0, 1.0 / fps],
        "observation.images.cam0": [0.0, 1.0 / fps],
    }

    standard = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
        video_backend="pyav",
    )
    fast = FastLeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
        lerobot_backend="parquet",
        lerobot_checks="all",
        video_backend="pyav",
    )

    assert len(standard) == len(fast)
    for idx in range(len(standard)):
        _compare_items(standard[idx], fast[idx])


def test_fast_lerobot_include_columns_filters_unneeded_features(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_OOB", "error")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1")
    repo_id, root, _fps = _build_tiny_dataset(
        tmp_path,
        repo_id="test_fast_lerobot_projection",
        use_videos=False,
    )

    fast = FastLeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,
        lerobot_backend="parquet",
        lerobot_checks="none",
        include_columns=["actions", "observation.state"],
    )

    item = fast[0]
    assert "actions" in item
    assert "observation.state" in item
    assert "observation.images.cam0" not in item
    assert "timestamp" in item
    assert "task_index" in item


def _tamper_episode_length(root: Path, episode_index: int, new_length: int) -> None:
    episodes_path = root / "meta" / "episodes.jsonl"
    lines = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["episode_index"] == episode_index:
                item["length"] = new_length
            lines.append(json.dumps(item))
    with episodes_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _inject_inconsistent_unused_feature_stats(root: Path) -> None:
    """Inject per-episode shape mismatch into an unused observation stats key."""
    stats_path = root / "meta" / "episodes_stats.jsonl"
    updated_lines: list[str] = []
    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ep_idx = int(item["episode_index"])
            dim = 2 if ep_idx % 2 == 0 else 3
            item["stats"]["observation.wrist"] = {
                "min": [0.0] * dim,
                "max": [1.0] * dim,
                "mean": [0.5] * dim,
                "std": [0.1] * dim,
                "count": [3],
            }
            updated_lines.append(json.dumps(item))
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines))
        f.write("\n")


def _parquet_total_rows(root: Path, repo_id: str, episodes: list[int]) -> int:
    meta = FastLeRobotDatasetMetadata(repo_id=repo_id, root=root)
    total = 0
    for ep_idx in episodes:
        path = root / meta.get_data_file_path(ep_idx)
        with pq.ParquetFile(path) as parquet:
            total += parquet.metadata.num_rows
    return total


def test_fast_lerobot_loads_when_unused_observation_stats_shape_differs_by_episode(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_OOB", "error")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1")
    repo_id, root, _fps = _build_tiny_dataset(
        tmp_path,
        repo_id="test_fast_lerobot_inconsistent_unused_stats",
        use_videos=False,
    )
    root_path = Path(root)
    _inject_inconsistent_unused_feature_stats(root_path)

    with pytest.raises(ValueError, match="all input arrays must have the same shape"):
        _ = LeRobotDatasetMetadata(repo_id=repo_id, root=root_path)

    fast = FastLeRobotDataset(
        repo_id=repo_id,
        root=root_path,
        download_videos=False,
        lerobot_backend="parquet",
        lerobot_checks="none",
        include_columns=["actions", "observation.state", "task_index", "timestamp"],
    )
    item = fast[0]
    assert "actions" in item
    assert "observation.state" in item
    assert "observation.wrist" not in item
    assert item["actions"].shape[-1] == 2


def test_fast_lerobot_uses_parquet_lengths(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_OOB", "error")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1")

    repo_id, root, _fps = _build_tiny_dataset(
        tmp_path,
        repo_id="test_fast_lerobot_lengths",
        use_videos=False,
    )
    root_path = Path(root)
    episodes = [0, 1]
    _tamper_episode_length(root_path, episode_index=0, new_length=999)

    fast = FastLeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,
        lerobot_backend="parquet",
        lerobot_checks="none",
    )
    actual_total = _parquet_total_rows(root_path, repo_id, episodes)
    assert len(fast) == actual_total
    for idx in range(len(fast)):
        _ = fast[idx]


def test_fast_lerobot_pickling_rebuilds_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_OOB", "error")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1")
    repo_id, root, _fps = _build_tiny_dataset(
        tmp_path,
        repo_id="test_fast_lerobot_pickle",
        use_videos=False,
    )

    fast = FastLeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,
        lerobot_backend="parquet",
        lerobot_checks="none",
    )

    restored = pickle.loads(pickle.dumps(fast))
    assert len(restored) == len(fast)
    _compare_items(fast[0], restored[0])


def test_fast_lerobot_setstate_without_include_columns_is_backward_compatible(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_OOB", "error")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1")
    repo_id, root, _fps = _build_tiny_dataset(
        tmp_path,
        repo_id="test_fast_lerobot_backward_include_columns",
        use_videos=False,
    )

    fast = FastLeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,
        lerobot_backend="parquet",
        lerobot_checks="none",
        include_columns=["actions", "observation.state"],
    )
    state = fast.__getstate__()
    state.pop("_include_columns", None)

    restored = FastLeRobotDataset.__new__(FastLeRobotDataset)
    restored.__setstate__(state)
    assert hasattr(restored, "_include_columns")
    assert restored._include_columns is None
    item = restored[0]
    assert "actions" in item
    assert "observation.state" in item


def test_fast_lerobot_row_group_cache_lru(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_OOB", "error")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1")
    monkeypatch.setenv("OPENPI_FAST_LEROBOT_ROW_GROUP_CACHE", "1")
    repo_id, root, _fps = _build_tiny_dataset(
        tmp_path,
        repo_id="test_fast_lerobot_cache",
        use_videos=False,
    )

    fast = FastLeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,
        lerobot_backend="parquet",
        lerobot_checks="none",
    )

    for idx in range(len(fast)):
        _ = fast[idx]
    assert len(fast._row_group_cache) <= 1
