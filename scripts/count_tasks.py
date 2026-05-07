#!/usr/bin/env python3
"""Count PA and short-horizon-task stats from parquet and episodes metadata."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")


def _read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    if not path.exists():
        return items
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _load_task_mapping(tasks_path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not tasks_path.exists():
        return mapping
    with tasks_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "task_index" not in item or "task" not in item:
                continue
            mapping[int(item["task_index"])] = str(item["task"])
    return mapping


def _tasks_from_episode(episode: dict, task_mapping: dict[int, str]) -> list[str]:
    tasks = episode.get("tasks", [])
    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(tasks, list):
        extracted: list[str] = []
        for task in tasks:
            if task is None:
                continue
            value = str(task).strip()
            if value:
                extracted.append(value)
        if extracted:
            return extracted

    task_index = episode.get("task_index")
    if isinstance(task_index, int):
        return [task_mapping.get(task_index, f"<missing task_index {task_index}>")]
    if isinstance(task_index, list):
        extracted = []
        for idx in task_index:
            if isinstance(idx, int):
                extracted.append(task_mapping.get(idx, f"<missing task_index {idx}>"))
        if extracted:
            return extracted

    return []


def _count_tasks_from_episodes(episodes: Iterable[dict], task_mapping: dict[int, str]) -> Counter:
    task_counter: Counter = Counter()
    for episode in episodes:
        tasks = _tasks_from_episode(episode, task_mapping)
        if tasks:
            task_counter.update(tasks)
        else:
            task_counter["<missing tasks>"] += 1
    return task_counter


def _read_task_indices(parquet_path: Path) -> list[int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pyarrow is required to read parquet files. "
            "Run inside the uv environment where lerobot is installed."
        ) from exc

    table = pq.read_table(parquet_path, columns=["task_index"], memory_map=True)
    if "task_index" not in table.column_names:
        raise KeyError(f"task_index column missing in {parquet_path}")
    values = table["task_index"].to_pylist()
    return [int(v) for v in values if v is not None]


def _count_tasks_from_parquet(dataset_root: Path, parquet_files: Iterable[Path]) -> Counter:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    task_mapping = _load_task_mapping(tasks_path)

    if not task_mapping:
        print(f"[WARN] tasks.jsonl not found or empty: {tasks_path}", file=sys.stderr)
        print("[WARN] Falling back to placeholder labels for task_index.", file=sys.stderr)

    task_counter: Counter = Counter()
    missing_task_indices: set[int] = set()

    for parquet_path in parquet_files:
        task_indices = _read_task_indices(parquet_path)
        if not task_indices:
            continue
        unique_indices = set(task_indices)
        if len(unique_indices) > 1:
            print(
                f"Warning: multiple task_index values found in {parquet_path}; counting per row.",
                file=sys.stderr,
            )
            indices_to_count = task_indices
        else:
            indices_to_count = list(unique_indices)

        for task_index in indices_to_count:
            task_name = task_mapping.get(task_index)
            if task_name is None:
                missing_task_indices.add(task_index)
                task_name = f"<missing task_index {task_index}>"
            task_counter[task_name] += 1

    if missing_task_indices:
        print(
            f"Warning: task_index not found in tasks.jsonl: {sorted(missing_task_indices)}",
            file=sys.stderr,
        )

    return task_counter


def _short_horizon_task_from_episode(episode: dict) -> str | None:
    value = episode.get("short_horizon_task")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _primitive_actions_from_episode(episode: dict, fallback_pa_tasks: list[str]) -> list[str]:
    primitive_actions = episode.get("primitive_action")
    if isinstance(primitive_actions, str):
        value = primitive_actions.strip()
        if value:
            return [value]
    if isinstance(primitive_actions, list):
        values: list[str] = []
        for item in primitive_actions:
            if item is None:
                continue
            value = str(item).strip()
            if value:
                values.append(value)
        if values:
            return values
    return fallback_pa_tasks


def _collect_short_horizon_stats(
    episodes: Iterable[dict],
    task_mapping: dict[int, str],
) -> tuple[Counter, dict[str, set[str]], dict[str, set[str]]]:
    short_counts: Counter = Counter()
    short_to_primitive: dict[str, set[str]] = defaultdict(set)
    primitive_to_short: dict[str, set[str]] = defaultdict(set)

    for episode in episodes:
        short_task = _short_horizon_task_from_episode(episode) or "<missing short_horizon_task>"
        pa_tasks = _tasks_from_episode(episode, task_mapping)
        primitive_actions = _primitive_actions_from_episode(episode, pa_tasks)

        short_counts[short_task] += 1
        for primitive_action in primitive_actions:
            short_to_primitive[short_task].add(primitive_action)
            primitive_to_short[primitive_action].add(short_task)

    return short_counts, dict(short_to_primitive), dict(primitive_to_short)


def _print_counter_table(title: str, label: str, counts: Counter) -> None:
    print(f"\n### {title}\n")
    print(f"| {label} | Count |")
    print("|---|---:|")
    for item, count in counts.most_common():
        print(f"| {_escape_md(item)} | {count} |")
    print(f"| **Total** | **{sum(counts.values())}** |")
    print(f"| **Unique** | **{len(counts)}** |")


def _print_short_horizon_table(short_counts: Counter, short_to_primitive: dict[str, set[str]]) -> None:
    print("\n### Short Horizon Task Stats\n")
    print("| Short Horizon Task | Episodes | Primitive Actions | Episodes / Primitive Action |")
    print("|---|---:|---:|---:|")

    ratios: list[float] = []
    for short_task, count in short_counts.most_common():
        primitive_count = len(short_to_primitive.get(short_task, set()))
        if primitive_count > 0:
            ratio = count / primitive_count
            ratio_text = f"{ratio:.3f}"
            ratios.append(ratio)
        else:
            ratio_text = "N/A"
        print(f"| {_escape_md(short_task)} | {count} | {primitive_count} | {ratio_text} |")

    print(f"| **Total episodes** | **{sum(short_counts.values())}** |  |  |")
    print(f"| **Unique short horizon tasks** | **{len(short_counts)}** |  |  |")
    if ratios:
        print(f"| **Average episodes / primitive action** |  |  | **{sum(ratios) / len(ratios):.3f}** |")


def _print_short_per_primitive_table(primitive_to_short: dict[str, set[str]]) -> None:
    print("\n### Short Horizon Tasks Per Primitive Action\n")
    print("| Primitive Action | Unique Short Horizon Tasks |")
    print("|---|---:|")

    counts: list[int] = []
    rows = sorted(primitive_to_short.items(), key=lambda item: (-len(item[1]), item[0]))
    for primitive_action, short_tasks in rows:
        num_short_tasks = len(short_tasks)
        counts.append(num_short_tasks)
        print(f"| {_escape_md(primitive_action)} | {num_short_tasks} |")

    if counts:
        print(f"| **Average short horizon tasks / primitive action** | **{sum(counts) / len(counts):.3f}** |")
    print(f"| **Unique primitive actions** | **{len(primitive_to_short)}** |")


def _parse_args() -> argparse.Namespace:
    default_root = Path(__file__).parent.parent / "datasets" / "test_dataset"
    parser = argparse.ArgumentParser(
        description="Count Primitive Actions and Short Horizon Tasks from a LeRobot dataset."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=default_root,
        help=f"Dataset root directory (default: {default_root})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root
    meta_root = dataset_root / "meta"
    episodes_path = meta_root / "episodes.jsonl"
    tasks_path = meta_root / "tasks.jsonl"

    task_mapping = _load_task_mapping(tasks_path)
    episodes = _read_jsonl(episodes_path)

    parquet_files = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if parquet_files:
        print(f"[INFO] Using parquet task_index for PA count from: {dataset_root / 'data'}")
        pa_counts = _count_tasks_from_parquet(dataset_root, parquet_files)
    elif episodes:
        print(f"[INFO] No parquet files found; using episodes metadata for PA count: {episodes_path}")
        pa_counts = _count_tasks_from_episodes(episodes, task_mapping)
    else:
        print(f"Error: File not found: {episodes_path}", file=sys.stderr)
        sys.exit(1)

    _print_counter_table("Primitive Action (PA) Count", "Primitive Action (PA)", pa_counts)

    if not episodes:
        print(f"\n[WARN] episodes.jsonl not found; skipping short horizon stats: {episodes_path}", file=sys.stderr)
        return

    short_counts, short_to_primitive, primitive_to_short = _collect_short_horizon_stats(episodes, task_mapping)
    _print_short_horizon_table(short_counts, short_to_primitive)
    _print_short_per_primitive_table(primitive_to_short)


if __name__ == "__main__":
    main()
