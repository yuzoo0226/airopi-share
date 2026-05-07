from __future__ import annotations

import bisect
import collections
import json
import logging
import os
from pathlib import Path
import socket
import time
from typing import Callable, Literal, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.utils

from lerobot.common.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
)
from lerobot.common.datasets.video_utils import decode_video_frames
from lerobot.common.datasets.utils import (
    check_delta_timestamps,
    check_timestamps_sync,
    get_delta_indices,
    get_episode_data_index,
    get_safe_version,
)
from openpi.training.fast_lerobot_metadata import FastLeRobotDatasetMetadata


class FastLeRobotDataset(LeRobotDataset):
    """LeRobotDataset variant that skips some heavy checks and groups queries."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        *,
        lerobot_backend: Literal["parquet", "hf"] = "parquet",
        lerobot_checks: Literal["all", "init_only", "none"] = "none",
        include_columns: Sequence[str] | None = None,
    ):
        """Initialize a fast dataset wrapper with optional parquet-only loading."""
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        requested_episodes = episodes
        init_episodes = episodes
        self._episodes_was_none = episodes is None
        if isinstance(init_episodes, list):
            init_episodes = init_episodes[:1]
        elif init_episodes is None:
            init_episodes = [0]
        if init_episodes != requested_episodes:
            logging.info("Using only episodes %s to initialize FastLeRobotDataset.", init_episodes)
        if root is None:
            from lerobot.common.constants import HF_LEROBOT_HOME

            self.root = HF_LEROBOT_HOME / repo_id
        else:
            self.root = Path(root)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = init_episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else None
        if self.video_backend is None:
            from lerobot.common.datasets.video_utils import get_safe_default_codec

            self.video_backend = get_safe_default_codec()
        self.delta_indices = None
        if lerobot_backend not in ("parquet", "hf"):
            raise ValueError(f"Unsupported lerobot_backend: {lerobot_backend}")
        if lerobot_checks not in ("all", "init_only", "none"):
            raise ValueError(f"Unsupported lerobot_checks: {lerobot_checks}")
        self._lerobot_backend = lerobot_backend
        self._lerobot_checks = lerobot_checks
        self._include_columns: set[str] | None = None
        if include_columns is not None:
            projected = {str(col) for col in include_columns if col}
            # Always keep core bookkeeping columns required by __getitem__.
            projected.update({"timestamp", "task_index"})
            self._include_columns = projected
        self._lazy_hf_dataset = lerobot_backend == "parquet"
        self._skip_file_check = lerobot_checks == "none"
        self._skip_timestamp_sync = lerobot_checks in ("init_only", "none")
        self._download_videos = download_videos
        self._force_cache_sync = force_cache_sync
        self._episode_from_np: np.ndarray | None = None
        self._episode_to_np: np.ndarray | None = None
        self._episode_index_to_data_idx: dict[int, int] | None = None
        self._row_group_cache: collections.OrderedDict[
            str, tuple[list[int], list[int], int]
        ] = collections.OrderedDict()
        self._row_group_cache_max = int(
            os.environ.get("OPENPI_FAST_LEROBOT_ROW_GROUP_CACHE", "4096")
        )
        self._warned_oob_index = False
        self._warned_oob_parquet: set[str] = set()
        self._oob_policy = os.environ.get("OPENPI_FAST_LEROBOT_OOB", "error")
        if self._oob_policy not in ("wrap", "error"):
            raise ValueError(f"Unsupported OPENPI_FAST_LEROBOT_OOB: {self._oob_policy}")
        self._use_parquet_lengths = (
            os.environ.get("OPENPI_FAST_LEROBOT_USE_PARQUET_LENGTHS", "1") == "1"
        )
        self._parquet_open_retries = int(os.environ.get("OPENPI_PARQUET_OPEN_RETRIES", "3"))
        self._parquet_open_backoff_sec = float(
            os.environ.get("OPENPI_PARQUET_OPEN_BACKOFF_SEC", "0.5")
        )
        self._skip_broken_episodes = (
            os.environ.get("OPENPI_SKIP_BROKEN_EPISODES", "1") == "1"
        )
        self._skip_broken_video_episodes = (
            os.environ.get("OPENPI_SKIP_BROKEN_VIDEO_EPISODES", "1") == "1"
        )
        self._skipped_broken_episodes: list[dict[str, str | int]] = []
        self._skip_broken_episodes_log_path = os.environ.get(
            "OPENPI_SKIP_BROKEN_EPISODES_LOG",
            str(self.root / "meta" / "openpi_skipped_episodes.jsonl"),
        )
        self._known_skipped_episode_indices = self._load_known_skipped_episode_indices()
        self._runtime_skipped_episode_indices: set[int] = set(self._known_skipped_episode_indices)
        self._getitem_recursion_depth = 0
        self._getitem_max_recursion = int(os.environ.get("OPENPI_GETITEM_MAX_RECURSION", "128"))
        if self._parquet_open_retries <= 0:
            raise ValueError(
                f"OPENPI_PARQUET_OPEN_RETRIES must be > 0, got {self._parquet_open_retries}."
            )
        if self._parquet_open_backoff_sec < 0:
            raise ValueError(
                "OPENPI_PARQUET_OPEN_BACKOFF_SEC must be >= 0, "
                f"got {self._parquet_open_backoff_sec}."
            )

        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        self.meta = FastLeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        # if requested_episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
        #     episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in requested_episodes]
        #     self.stats = aggregate_stats(episodes_stats)

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        self._refresh_episode_index_cache()

        self.hf_dataset = None
        if not self._lazy_hf_dataset:
            self._ensure_hf_dataset()

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        if init_episodes != requested_episodes:
            logging.info("Loading all episodes ...")
            self.episodes = requested_episodes
            self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
            self._refresh_episode_index_cache()
            if not self._lazy_hf_dataset:
                self.hf_dataset = None
                self._ensure_hf_dataset()
            logging.info("Loaded all episodes.")

        if self._lazy_hf_dataset:
            self._ensure_parquet_data()
            self._initialize_parquet_index()
            if self._skip_timestamp_sync:
                logging.info("Skipping LeRobot timestamp sync check.")
            else:
                self._run_timestamp_sync_check_parquet()
        else:
            if self._skip_timestamp_sync:
                logging.info("Skipping LeRobot timestamp sync check.")

        self._debug_dataset = os.environ.get("OPENPI_DEBUG_DATASET") == "1"
        if self._debug_dataset:
            def _fmt_eps(eps: list[int] | None) -> str:
                """Format episode lists for concise debug logging."""
                if eps is None:
                    return "all"
                if not eps:
                    return "[]"
                if len(eps) <= 6:
                    return str(eps)
                return f"{eps[:3]}...{eps[-3:]}"

            if self.episode_data_index is None or self.episode_data_index["to"].numel() == 0:
                dataset_size = 0
            else:
                dataset_size = int(self.episode_data_index["to"][-1].item())
            selected_count = (
                self.meta.total_episodes if self._episodes_was_none else len(self.episodes)
            )
            logging.info(
                "FastLeRobotDataset debug: backend=%s checks=%s requested_episodes=%s selected_episodes=%s selected_count=%s dataset_size=%s meta_total_frames=%s meta_total_episodes=%s",
                self._lerobot_backend,
                self._lerobot_checks,
                _fmt_eps(requested_episodes),
                _fmt_eps(self.episodes),
                selected_count,
                dataset_size,
                self.meta.total_frames,
                self.meta.total_episodes,
            )

        self._profile_enabled = os.environ.get("OPENPI_PROFILE_DATASET") == "1"
        self._profile_every = int(os.environ.get("OPENPI_PROFILE_DATASET_EVERY", "200"))
        self._profile_samples = 0
        self._profile_sums: dict[str, float] = {}

    @property
    def num_frames(self) -> int:
        """Return the number of frames available in the selected episodes."""
        if self.hf_dataset is not None:
            return len(self.hf_dataset)
        if self.episode_data_index is None or self.episode_data_index["to"].numel() == 0:
            return 0
        return int(self.episode_data_index["to"][-1].item())

    def __getitem__(self, idx: int) -> dict:
        """Fetch a sample and optional delta/video queries by index."""
        self._getitem_recursion_depth += 1
        if self._getitem_recursion_depth > self._getitem_max_recursion:
            self._getitem_recursion_depth -= 1
            raise RuntimeError(
                "Exceeded max __getitem__ recursion while skipping broken episodes. "
                f"limit={self._getitem_max_recursion}"
            )

        t_start = time.perf_counter() if self._profile_enabled else 0.0
        try:
            idx = int(idx)
            use_hf_dataset = self.hf_dataset is not None or not self._lazy_hf_dataset
            if use_hf_dataset:
                t0 = time.perf_counter() if self._profile_enabled else 0.0
                self._ensure_hf_dataset()
                dataset_size = len(self.hf_dataset)
                item = self.hf_dataset[idx]
                ep_idx_video = int(item["episode_index"].item())
                if self.episodes is None:
                    ep_idx_data = ep_idx_video
                elif self._episode_index_to_data_idx is None:
                    raise IndexError("Episode index map is empty; no episodes available.")
                else:
                    try:
                        ep_idx_data = self._episode_index_to_data_idx[ep_idx_video]
                    except KeyError as exc:
                        raise IndexError(
                            f"Episode index {ep_idx_video} is not in the selected episodes."
                        ) from exc
                if self._profile_enabled:
                    self._profile_add("hf_get", time.perf_counter() - t0)
            else:
                if self._episode_to_np is None or self._episode_from_np is None:
                    self._refresh_episode_index_cache()
                if self._episode_to_np is None or self._episode_to_np.size == 0:
                    raise IndexError("Episode index cache is empty; no episodes available.")
                dataset_size = int(self._episode_to_np[-1])
                if idx < 0:
                    idx = dataset_size + idx
                if idx < 0 or idx >= dataset_size:
                    if self._oob_policy == "error":
                        raise IndexError(
                            f"Sample index {idx} is out of bounds for dataset size {dataset_size}."
                        )
                    if not self._warned_oob_index:
                        selected_count = (
                            self.meta.total_episodes if self._episodes_was_none else len(self.episodes)
                        )
                        logging.warning(
                            "Sample index %s is out of bounds for dataset size %s; wrapping index. meta_total_frames=%s meta_total_episodes=%s selected_count=%s root=%s",
                            idx,
                            dataset_size,
                            self.meta.total_frames,
                            self.meta.total_episodes,
                            selected_count,
                            self.root,
                        )
                        self._warned_oob_index = True
                    idx = idx % dataset_size
                ep_idx_data = int(np.searchsorted(self._episode_to_np, idx, side="right"))
                if ep_idx_data >= self._episode_to_np.size:
                    raise IndexError(
                        f"Sample index {idx} maps to invalid episode index {ep_idx_data}."
                    )
                t0 = time.perf_counter() if self._profile_enabled else 0.0
                ep_idx_video = int(self.episodes[ep_idx_data] if self.episodes is not None else ep_idx_data)
                item = self._get_item_from_parquet(ep_idx_data, idx)
                if self._profile_enabled:
                    self._profile_add("parquet_get", time.perf_counter() - t0)

            if ep_idx_video in self._runtime_skipped_episode_indices and dataset_size > 1:
                # Jump to the first frame of the next episode instead of idx+1,
                # so we don't recurse through every frame of the broken episode.
                if self._episode_to_np is not None and ep_idx_data < len(self._episode_to_np):
                    fallback_idx = int(self._episode_to_np[ep_idx_data]) % dataset_size
                else:
                    fallback_idx = (idx + 1) % dataset_size
                if fallback_idx != idx:
                    return self.__getitem__(fallback_idx)

            query_indices = None
            if self.delta_indices is not None:
                t0 = time.perf_counter() if self._profile_enabled else 0.0
                query_indices, padding = self._get_query_indices(idx, ep_idx_data)
                query_result = self._query_hf_dataset(query_indices, ep_idx_data)
                item = {**item, **padding, **query_result}
                if self._profile_enabled:
                    self._profile_add("delta_query", time.perf_counter() - t0)

            if len(self.meta.video_keys) > 0:
                t0 = time.perf_counter() if self._profile_enabled else 0.0
                current_ts = item["timestamp"].item()
                query_timestamps = self._get_query_timestamps(current_ts, query_indices, ep_idx_data)
                try:
                    video_frames = self._query_videos(query_timestamps, ep_idx_video)
                except Exception as exc:
                    if not self._skip_broken_video_episodes:
                        raise
                    if ep_idx_video not in self._runtime_skipped_episode_indices:
                        self._runtime_skipped_episode_indices.add(ep_idx_video)
                        self._record_skipped_broken_episode(
                            ep_idx=ep_idx_video,
                            path=self.root / self.meta.get_video_file_path(ep_idx_video, self.meta.video_keys[0]),
                            stage="video_decode",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                        self._flush_skipped_broken_episodes_log()
                    if dataset_size > 1:
                        # Jump to the first frame of the next episode instead of idx+1.
                        if self._episode_to_np is not None and ep_idx_data < len(self._episode_to_np):
                            fallback_idx = int(self._episode_to_np[ep_idx_data]) % dataset_size
                        else:
                            fallback_idx = (idx + 1) % dataset_size
                        if fallback_idx != idx:
                            return self.__getitem__(fallback_idx)
                    raise
                item = {**video_frames, **item}
                if self._profile_enabled:
                    self._profile_add("video_decode", time.perf_counter() - t0)

            if self.image_transforms is not None:
                t0 = time.perf_counter() if self._profile_enabled else 0.0
                for cam in self.meta.camera_keys:
                    item[cam] = self.image_transforms(item[cam])
                if self._profile_enabled:
                    self._profile_add("image_tf", time.perf_counter() - t0)

            task_idx = item["task_index"].item()
            item["task"] = self.meta.tasks[task_idx]
            if self._profile_enabled:
                self._profile_add("total", time.perf_counter() - t_start)
                self._profile_maybe_log()
            return item
        finally:
            self._getitem_recursion_depth -= 1

    def _get_item_from_parquet(self, ep_idx_data: int, idx: int) -> dict:
        """Read a single item directly from a parquet episode file."""
        first = self.episode_data_index["from"][ep_idx_data].item()
        if self.episodes is not None:
            ep_idx_data = self.episodes[ep_idx_data]

        path = str(self.root / self.meta.get_data_file_path(ep_idx_data))
        rel_idx = idx - first
        columns = [key for key in self.meta.features if key not in self.meta.video_keys]
        include_columns = getattr(self, "_include_columns", None)
        if include_columns is not None:
            projected_columns = [key for key in columns if key in include_columns]
            if projected_columns:
                columns = projected_columns
        row = self._read_parquet_row(path, columns, rel_idx)
        return row

    def _query_hf_dataset(self, query_indices: dict[str, list[int]], ep_idx_data: int) -> dict:
        """Query multiple columns for delta indices from parquet files."""
        first = self.episode_data_index["from"][ep_idx_data].item()
        if self.episodes is not None:
            ep_idx_data = self.episodes[ep_idx_data]

        path = str(self.root / self.meta.get_data_file_path(ep_idx_data))

        query_indices = {
            key: q_idx for key, q_idx in query_indices.items() if key not in self.meta.video_keys
        }
        return self._select_hf_dataset_multi(query_indices, first, path)

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None,
        ep_idx_data: int,
    ) -> dict[str, list[float]]:
        """Resolve timestamps to fetch for video keys."""
        first = self.episode_data_index["from"][ep_idx_data].item()
        if self.episodes is not None:
            ep_idx_data = self.episodes[ep_idx_data]

        path = str(self.root / self.meta.get_data_file_path(ep_idx_data))

        query_timestamps = {}
        timestamp_map: dict[int, float] = {}
        if query_indices is not None:
            indexed = {key: query_indices[key] for key in self.meta.video_keys if key in query_indices}
            if indexed:
                union_indices = sorted({idx for q_idx in indexed.values() for idx in q_idx})
                timestamps = self._select_hf_dataset("timestamp", union_indices, first, path).tolist()
                timestamp_map = dict(zip(union_indices, timestamps))

        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                query_timestamps[key] = [timestamp_map[idx] for idx in query_indices[key]]
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Decode video frames for the requested timestamps."""
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            try:
                frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            except Exception as exc:  # noqa: BLE001
                logging.error(
                    "Failed to decode video '%s' (episode=%s key=%s backend=%s): %s",
                    video_path,
                    ep_idx,
                    vid_key,
                    self.video_backend,
                    exc,
                )
                raise
            item[vid_key] = frames.squeeze(0)

        return item

    def _get_row_group_boundaries(
        self, parquet: pq.ParquetFile
    ) -> tuple[list[int], list[int], int]:
        """Return row group start indices, sizes, and total row count."""
        row_starts = []
        row_counts = []
        row_start = 0
        for rg_idx in range(parquet.num_row_groups):
            row_starts.append(row_start)
            rg_rows = parquet.metadata.row_group(rg_idx).num_rows
            row_counts.append(rg_rows)
            row_start += rg_rows
        return row_starts, row_counts, row_start

    def _get_row_group_boundaries_cached(
        self, path: str, parquet: pq.ParquetFile
    ) -> tuple[list[int], list[int], int]:
        """Cache row group boundaries to avoid repeated parquet metadata scans."""
        cached = self._row_group_cache.get(path)
        if cached is not None:
            self._row_group_cache.move_to_end(path)
            return cached
        row_starts, row_counts, total_rows = self._get_row_group_boundaries(parquet)
        if self._row_group_cache_max > 0:
            self._row_group_cache[path] = (row_starts, row_counts, total_rows)
            self._row_group_cache.move_to_end(path)
            while len(self._row_group_cache) > self._row_group_cache_max:
                self._row_group_cache.popitem(last=False)
        return row_starts, row_counts, total_rows

    def _select_row_groups(self, row_starts: list[int], rel_indices: list[int]) -> list[int]:
        """Pick row groups that contain the requested relative indices."""
        selected = set()
        for rel_idx in rel_indices:
            rg_idx = bisect.bisect_right(row_starts, rel_idx) - 1
            selected.add(rg_idx)
        return sorted(selected)

    def _compute_row_group_offsets(
        self, selected_row_groups: list[int], row_counts: list[int]
    ) -> dict[int, int]:
        """Compute offsets into the concatenated table for selected row groups."""
        offsets = {}
        offset = 0
        for rg_idx in selected_row_groups:
            offsets[rg_idx] = offset
            offset += row_counts[rg_idx]
        return offsets

    def _map_rel_indices_to_table_indices(
        self,
        rel_indices: list[int],
        row_starts: list[int],
        row_group_offsets: dict[int, int],
    ) -> list[int]:
        """Map file-relative indices to indices within the read table."""
        table_indices = []
        for rel_idx in rel_indices:
            rg_idx = bisect.bisect_right(row_starts, rel_idx) - 1
            table_indices.append(row_group_offsets[rg_idx] + (rel_idx - row_starts[rg_idx]))
        return table_indices

    def _select_hf_dataset_multi(
        self, query_indices: dict[str, list[int]], first: int, path: str
    ) -> dict[str, torch.Tensor]:
        """Batch-select multiple columns from parquet with shared row group reads."""
        if not query_indices:
            return {}

        empty = {key: torch.tensor([]) for key, q_idx in query_indices.items() if not q_idx}
        non_empty = {key: q_idx for key, q_idx in query_indices.items() if q_idx}
        if not non_empty:
            return empty

        rel_indices_union = sorted({idx - first for q_idx in non_empty.values() for idx in q_idx})
        with self._open_parquet_file(path) as parquet:
            row_starts, row_counts, total_rows = self._get_row_group_boundaries_cached(path, parquet)
            rel_indices_union = sorted(
                set(self._normalize_rel_indices(rel_indices_union, total_rows, path))
            )
            selected_row_groups = self._select_row_groups(row_starts, rel_indices_union)
            table = parquet.read_row_groups(selected_row_groups, columns=list(non_empty.keys()))

        row_group_offsets = self._compute_row_group_offsets(selected_row_groups, row_counts)
        out: dict[str, torch.Tensor] = {}
        for key, q_idx in non_empty.items():
            rel_indices = [idx - first for idx in q_idx]
            rel_indices = self._normalize_rel_indices(rel_indices, total_rows, path)
            table_indices = self._map_rel_indices_to_table_indices(
                rel_indices, row_starts, row_group_offsets
            )
            values = table[key].take(pa.array(table_indices)).to_pylist()
            out[key] = self._coerce_parquet_values(key, values)
        out.update(empty)
        return out

    def _select_hf_dataset(self, key: str, q_idx: list[int], first: int, path: str) -> torch.Tensor:
        """Select a single column from a parquet episode file."""
        if not q_idx:
            return torch.tensor([])
        rel_indices = [idx - first for idx in q_idx]
        with self._open_parquet_file(path) as parquet:
            row_starts, row_counts, total_rows = self._get_row_group_boundaries_cached(path, parquet)
            rel_indices = self._normalize_rel_indices(rel_indices, total_rows, path)
            selected_row_groups = self._select_row_groups(row_starts, rel_indices)
            table = parquet.read_row_groups(selected_row_groups, columns=[key])

        row_group_offsets = self._compute_row_group_offsets(selected_row_groups, row_counts)
        table_indices = self._map_rel_indices_to_table_indices(
            rel_indices, row_starts, row_group_offsets
        )
        values = table[key].take(pa.array(table_indices)).to_pylist()
        return self._coerce_parquet_values(key, values)

    def _read_parquet_row(self, path: str, columns: list[str], rel_idx: int) -> dict:
        """Read a single row from a parquet file for specific columns."""
        with self._open_parquet_file(path) as parquet:
            row_starts, _row_counts, total_rows = self._get_row_group_boundaries_cached(path, parquet)
            rel_idx = self._normalize_rel_indices([rel_idx], total_rows, path)[0]
            row_group_idx = bisect.bisect_right(row_starts, rel_idx) - 1
            rel_in_group = rel_idx - row_starts[row_group_idx]

            table = parquet.read_row_group(row_group_idx, columns=columns)
            row = table.slice(rel_in_group, 1)

        out: dict[str, object] = {}
        for key in columns:
            value = row[key].to_pylist()[0]
            out[key] = self._coerce_parquet_value(value)
        return out

    def _coerce_parquet_value(self, value: object) -> object:
        """Convert a parquet scalar value into a torch-friendly type."""
        if value is None or isinstance(value, (str, bytes)):
            return value
        if isinstance(value, dict):
            decoded = self._decode_parquet_image(value)
            if decoded is not None:
                return decoded
        if isinstance(value, list):
            return torch.tensor(value)
        return torch.tensor(value)

    def _coerce_parquet_values(self, key: str, values: list[object]) -> torch.Tensor:
        """Convert a list of parquet values into a tensor or stacked images."""
        if not values:
            return torch.tensor([])
        first = values[0]
        if isinstance(first, dict):
            if "bytes" in first or "path" in first:
                decoded = [self._decode_parquet_image(value) for value in values]
                if any(item is None for item in decoded):
                    raise TypeError(f"Unsupported parquet image payload for key '{key}'.")
                return torch.stack(decoded)
            raise TypeError(f"Unsupported parquet dict payload for key '{key}'.")
        return torch.tensor(values)

    def _normalize_rel_indices(
        self, rel_indices: list[int], total_rows: int, path: str
    ) -> list[int]:
        """Validate or wrap relative indices based on the OOB policy."""
        if total_rows <= 0:
            raise IndexError(f"Parquet file has no rows: {path}")
        if self._oob_policy == "error":
            for rel_idx in rel_indices:
                if rel_idx < 0 or rel_idx >= total_rows:
                    raise IndexError(f"Row index {rel_idx} is out of bounds for {path}")
            return rel_indices
        normalized = []
        had_oob = False
        for rel_idx in rel_indices:
            if rel_idx < 0 or rel_idx >= total_rows:
                had_oob = True
                rel_idx = rel_idx % total_rows
            normalized.append(rel_idx)
        if had_oob and path not in self._warned_oob_parquet:
            logging.warning(
                "Parquet file (%s) doesn't have required size. "
                "Wrapping indices to available rows (%s).",
                path,
                total_rows,
            )
            self._warned_oob_parquet.add(path)
        return normalized

    def _decode_parquet_image(self, value: dict) -> torch.Tensor | None:
        """Decode image payloads stored in parquet rows."""
        if "bytes" not in value and "path" not in value:
            return None
        try:
            from datasets import Image as DatasetImage
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                "Decoding image columns from parquet requires 'datasets' and 'torchvision'."
            ) from exc
        image = DatasetImage().decode_example(value)
        return transforms.ToTensor()(image)

    def _refresh_episode_index_cache(self) -> None:
        """Cache episode index tensors as numpy arrays and build lookup map."""
        if self.episode_data_index is None:
            self._episode_from_np = None
            self._episode_to_np = None
            self._episode_index_to_data_idx = None
            return
        self._episode_from_np = self.episode_data_index["from"].cpu().numpy()
        self._episode_to_np = self.episode_data_index["to"].cpu().numpy()
        if self.episodes is None:
            self._episode_index_to_data_idx = None
        else:
            self._episode_index_to_data_idx = {ep_idx: i for i, ep_idx in enumerate(self.episodes)}

    def __getstate__(self) -> dict:
        """Return a pickle-safe state with caches cleared."""
        state = self.__dict__.copy()
        # Rebuild caches in worker processes to avoid stale shared views.
        state["_episode_from_np"] = None
        state["_episode_to_np"] = None
        state["_episode_index_to_data_idx"] = None
        state["_row_group_cache"] = collections.OrderedDict()
        # Keep projection settings across worker process boundaries.
        state.setdefault("_include_columns", getattr(self, "_include_columns", None))
        # hf_dataset can be large; only drop it for parquet backend where it's unused.
        if self._lazy_hf_dataset:
            state["hf_dataset"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and rebuild derived caches."""
        self.__dict__.update(state)
        # Backward compatibility for objects pickled before _include_columns was introduced.
        if "_include_columns" not in self.__dict__:
            self._include_columns = None
        self._refresh_episode_index_cache()

    def _initialize_parquet_index(self) -> None:
        """Build episode index based on parquet lengths or metadata."""
        if not self._lazy_hf_dataset:
            return
        episodes = self._resolve_parquet_episodes()
        if not episodes:
            self.episode_data_index = None
            self.episodes = []
            self._refresh_episode_index_cache()
            return
        if self._use_parquet_lengths:
            episodes, lengths = self._get_parquet_lengths(episodes)
        else:
            episodes = self._filter_broken_episodes(episodes)
            lengths = self._get_meta_lengths(episodes)

        if not episodes:
            raise RuntimeError(
                "No usable episodes remain after filtering broken parquet episodes. "
                f"root={self.root} repo_id={self.repo_id}"
            )

        self.episodes = episodes
        self.episode_data_index = self._build_episode_index(lengths)
        self._refresh_episode_index_cache()
        self._flush_skipped_broken_episodes_log()

    def _resolve_parquet_episodes(self) -> list[int]:
        """Resolve the list of episode indices to load for parquet access."""
        if self.episodes is None:
            episodes = sorted(self.meta.episodes)
        else:
            episodes = list(self.episodes)

        if not self._skip_broken_episodes or not self._known_skipped_episode_indices:
            return episodes

        filtered = [ep for ep in episodes if ep not in self._known_skipped_episode_indices]
        skipped_known = len(episodes) - len(filtered)
        if skipped_known > 0:
            logging.info(
                "Pre-filtered %d episodes from skip cache before parquet scan.",
                skipped_known,
            )
        return filtered

    def _load_known_skipped_episode_indices(self) -> set[int]:
        """Load previously skipped episode indices from JSONL log file."""
        if not self._skip_broken_episodes:
            return set()

        log_path = Path(self._skip_broken_episodes_log_path)
        if not log_path.exists():
            return set()

        known: set[int] = set()
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue

                    row_repo = row.get("repo_id")
                    if row_repo not in (None, self.repo_id):
                        continue

                    ep_idx = row.get("episode_index")
                    if isinstance(ep_idx, int):
                        known.add(ep_idx)
                    elif isinstance(ep_idx, str) and ep_idx.isdigit():
                        known.add(int(ep_idx))
        except Exception as exc:
            logging.warning(
                "Failed to load skipped episode cache from %s: %s",
                log_path,
                exc,
            )
            return set()

        if known:
            logging.info(
                "Loaded %d known skipped episodes from %s",
                len(known),
                log_path,
            )
        return known

    def _open_parquet_file(self, path: str | Path) -> pq.ParquetFile:
        """Open a parquet file with retry and detailed failure logging."""
        path_str = str(path)
        host = os.uname().nodename
        realpath = os.path.realpath(path_str)
        last_exc: Exception | None = None

        for attempt in range(1, self._parquet_open_retries + 1):
            try:
                return pq.ParquetFile(path_str)
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    "Parquet open failed (attempt %s/%s): host=%s path=%s realpath=%s error=%s",
                    attempt,
                    self._parquet_open_retries,
                    host,
                    path_str,
                    realpath,
                    repr(exc),
                )
                if attempt < self._parquet_open_retries:
                    time.sleep(self._parquet_open_backoff_sec * (2 ** (attempt - 1)))

        message = (
            "Parquet open failed after retries: "
            f"host={host} path={path_str} realpath={realpath} "
            f"retries={self._parquet_open_retries} last_error={repr(last_exc)}"
        )
        logging.error(message)
        raise RuntimeError(message) from last_exc

    def _get_parquet_lengths(self, episodes: list[int]) -> tuple[list[int], list[int]]:
        """Read parquet metadata to get per-episode lengths.

        Returns:
            Tuple of (usable_episodes, lengths)
        """
        usable_episodes: list[int] = []
        lengths = []
        for ep_idx in episodes:
            path = self.root / self.meta.get_data_file_path(ep_idx)
            try:
                with self._open_parquet_file(path) as parquet:
                    lengths.append(parquet.metadata.num_rows)
                    usable_episodes.append(ep_idx)
            except Exception as exc:
                if not self._skip_broken_episodes:
                    raise
                self._record_skipped_broken_episode(
                    ep_idx=ep_idx,
                    path=path,
                    stage="length_scan",
                    reason=f"{type(exc).__name__}: {exc}",
                )

        return usable_episodes, lengths

    def _filter_broken_episodes(self, episodes: list[int]) -> list[int]:
        """Filter episodes whose parquet path is not a file."""
        if not self._skip_broken_episodes:
            return episodes

        usable_episodes: list[int] = []
        for ep_idx in episodes:
            path = self.root / self.meta.get_data_file_path(ep_idx)
            try:
                if path.is_file():
                    usable_episodes.append(ep_idx)
                    continue
                self._record_skipped_broken_episode(
                    ep_idx=ep_idx,
                    path=path,
                    stage="path_check",
                    reason=(
                        f"invalid_path_state exists={path.exists()} "
                        f"is_file={path.is_file()} is_dir={path.is_dir()}"
                    ),
                )
            except Exception as exc:
                self._record_skipped_broken_episode(
                    ep_idx=ep_idx,
                    path=path,
                    stage="path_check",
                    reason=f"{type(exc).__name__}: {exc}",
                )

        return usable_episodes

    def _record_skipped_broken_episode(
        self,
        *,
        ep_idx: int,
        path: str | Path,
        stage: str,
        reason: str,
    ) -> None:
        path_str = str(path)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "repo_id": self.repo_id,
            "root": str(self.root),
            "episode_index": int(ep_idx),
            "path": path_str,
            "realpath": os.path.realpath(path_str),
            "stage": stage,
            "reason": reason,
        }
        self._skipped_broken_episodes.append(entry)
        logging.warning(
            "Skipping broken episode: episode=%s stage=%s path=%s reason=%s",
            ep_idx,
            stage,
            path_str,
            reason,
        )

    def _flush_skipped_broken_episodes_log(self) -> None:
        if not self._skipped_broken_episodes:
            return

        log_path = Path(self._skip_broken_episodes_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            for entry in self._skipped_broken_episodes:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logging.warning(
            "Skipped %d broken episodes. Details appended to %s",
            len(self._skipped_broken_episodes),
            log_path,
        )

    def _get_meta_lengths(self, episodes: list[int]) -> list[int]:
        """Get per-episode lengths from metadata."""
        return [self.meta.episodes[ep_idx]["length"] for ep_idx in episodes]

    def _build_episode_index(self, lengths: list[int]) -> dict[str, torch.Tensor]:
        """Construct cumulative start/end indices from episode lengths."""
        cumulative = np.cumsum(np.array(lengths, dtype=np.int64))
        starts = np.concatenate(([0], cumulative[:-1]))
        return {
            "from": torch.from_numpy(starts),
            "to": torch.from_numpy(cumulative),
        }

    def _profile_add(self, key: str, dt: float) -> None:
        """Accumulate timing stats for profiling."""
        self._profile_sums[key] = self._profile_sums.get(key, 0.0) + dt

    def _profile_maybe_log(self) -> None:
        """Log averaged profiling stats at a fixed sampling interval."""
        self._profile_samples += 1
        if self._profile_samples % self._profile_every != 0:
            return
        parts = []
        for key in sorted(self._profile_sums):
            avg_ms = (self._profile_sums[key] / self._profile_samples) * 1000.0
            parts.append(f"{key}={avg_ms:.2f}ms")
        msg = f"FastLeRobotDataset profile avg ({self._profile_samples} samples): {' '.join(parts)}"
        logging.info("%s", msg)
        print(msg, flush=True)

    def _ensure_parquet_data(self) -> None:
        """Ensure parquet files exist locally, downloading if needed."""
        try:
            if self._force_cache_sync:
                raise FileNotFoundError
            if not self._skip_file_check:
                self._run_parquet_file_check()
        except (FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self._row_group_cache.clear()
            self.download_episodes(self._download_videos)

    def _run_parquet_file_check(self) -> None:
        """Validate the presence of required parquet files."""
        episodes = self.episodes if self.episodes is not None else range(self.meta.total_episodes)
        missing = []
        for ep_idx in episodes:
            path = self.root / self.meta.get_data_file_path(ep_idx)
            if not path.is_file():
                missing.append(path)
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(f"Missing parquet data files: {preview}")

    def _run_timestamp_sync_check_parquet(self) -> None:
        """Verify timestamps in parquet files match fps within tolerance."""
        if self.episode_data_index is None or self.episode_data_index["to"].numel() == 0:
            return
        episodes = self.episodes if self.episodes is not None else range(self.meta.total_episodes)
        timestamps_parts = []
        episode_indices_parts = []
        for ep_idx in episodes:
            path = self.root / self.meta.get_data_file_path(ep_idx)
            with self._open_parquet_file(path) as parquet:
                for batch in parquet.iter_batches(columns=["timestamp"]):
                    timestamps = batch.column(0).to_numpy()
                    timestamps_parts.append(timestamps)
                    episode_indices_parts.append(np.full(timestamps.shape, ep_idx, dtype=np.int64))
        if not timestamps_parts:
            return
        timestamps = np.concatenate(timestamps_parts)
        episode_indices = np.concatenate(episode_indices_parts)
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

    def _ensure_hf_dataset(self) -> None:
        """Load the Hugging Face dataset and run checks if configured."""
        if self.hf_dataset is not None:
            return
        try:
            if self._force_cache_sync:
                raise FileNotFoundError
            if not self._skip_file_check:
                assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(self._download_videos)
            self.hf_dataset = self.load_hf_dataset()
        if not self._skip_timestamp_sync:
            self._run_timestamp_sync_check()

    def _run_timestamp_sync_check(self) -> None:
        """Verify HF dataset timestamps match fps within tolerance."""
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)
