from __future__ import annotations

from pathlib import Path

import packaging.version

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    backward_compatible_episodes_stats,
    check_version_compatibility,
    get_safe_version,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
)


class FastLeRobotDatasetMetadata(LeRobotDatasetMetadata):
    """LeRobot metadata loader that defers expensive stats parsing."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self._episodes_stats = None
        self._stats = None
        self._stats_loaded = False

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self) -> None:
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        self._episodes_stats = None
        self._stats = None
        self._stats_loaded = False

    @property
    def episodes_stats(self) -> dict:
        if not self._stats_loaded:
            self._load_stats()
        return self._episodes_stats

    @property
    def stats(self) -> dict:
        if not self._stats_loaded:
            self._load_stats()
        return self._stats

    def _load_stats(self) -> None:
        if self._stats_loaded:
            return
        if self._version < packaging.version.parse("v2.1"):
            stats = load_stats(self.root)
            episodes_stats = backward_compatible_episodes_stats(stats, self.episodes)
        else:
            episodes_stats = load_episodes_stats(self.root)
            stats = aggregate_stats(list(episodes_stats.values()))
        self._episodes_stats = episodes_stats
        self._stats = stats
        self._stats_loaded = True
