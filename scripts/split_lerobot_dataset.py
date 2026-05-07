#!/usr/bin/env python3
"""Split a LeRobot dataset into train/val datasets by episode."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True))
            f.write("\n")


def _format_template(template: str, *, episode_index: int, chunk_size: int, video_key: str | None = None) -> str:
    episode_chunk = episode_index // chunk_size if chunk_size > 0 else 0
    values: dict[str, object] = {
        "episode_index": episode_index,
        "episode_chunk": episode_chunk,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return template.format(**values)


def _copy_tree_entry(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def _get_video_keys(info: dict) -> list[str]:
    features = info.get("features", {})
    keys = []
    for key, spec in features.items():
        if isinstance(spec, dict) and spec.get("dtype") == "video":
            keys.append(str(key))
    return keys


def _validate_outputs(train_out: Path, val_out: Path, *, overwrite: bool, skip_existing: bool) -> bool:
    existing = [p for p in (train_out, val_out) if p.exists()]
    if not existing:
        return True
    if overwrite:
        for p in existing:
            shutil.rmtree(p)
        return True
    if skip_existing:
        print(f"[INFO] Output already exists. Skipping split: {existing}")
        return False
    raise FileExistsError(f"Output already exists: {existing}. Use --overwrite or --skip-existing.")


def _load_episode_items(episodes_path: Path) -> list[dict]:
    episodes = _read_jsonl(episodes_path)
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {episodes_path}")
    return episodes


def _split_indices(episode_indices: list[int], train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = episode_indices[:]
    rng.shuffle(shuffled)
    split_at = int(len(shuffled) * train_fraction)
    train = sorted(shuffled[:split_at])
    val = sorted(shuffled[split_at:])
    return train, val


def _split_indices_stratified(
    episodes: list[dict],
    train_fraction: float,
    seed: int,
    task_mapping: dict[int, str],
    mode: str,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_task: dict[str, list[int]] = {}
    for item in episodes:
        idx = item.get("episode_index")
        if not isinstance(idx, int):
            continue
        tasks = _episode_tasks(item, task_mapping)
        if not tasks:
            label = "<missing tasks>"
        elif mode == "all":
            label = " | ".join(tasks)
        else:
            label = tasks[0]
        by_task.setdefault(label, []).append(idx)

    train: list[int] = []
    val: list[int] = []
    for label in sorted(by_task.keys()):
        indices = by_task[label]
        rng.shuffle(indices)
        split_at = int(len(indices) * train_fraction)
        train.extend(indices[:split_at])
        val.extend(indices[split_at:])

    return sorted(train), sorted(val)


def _episode_lookup(episodes: list[dict]) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for item in episodes:
        idx = item.get("episode_index")
        if isinstance(idx, int):
            lookup[idx] = item
    return lookup


def _stats_lookup(stats: list[dict]) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for item in stats:
        idx = item.get("episode_index")
        if isinstance(idx, int):
            lookup[idx] = item
    return lookup


def _load_task_mapping(tasks_path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not tasks_path.exists():
        return mapping
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "task_index" not in item or "task" not in item:
            continue
        mapping[int(item["task_index"])] = str(item["task"])
    return mapping


def _episode_tasks(item: dict, task_mapping: dict[int, str]) -> list[str]:
    tasks = item.get("tasks", [])
    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(tasks, list):
        return [str(t) for t in tasks if t]

    task_index = item.get("task_index")
    if isinstance(task_index, int):
        task_name = task_mapping.get(task_index, f"<missing task_index {task_index}>")
        return [task_name]
    if isinstance(task_index, list):
        task_names = []
        for idx in task_index:
            if isinstance(idx, int):
                task_names.append(task_mapping.get(idx, f"<missing task_index {idx}>"))
        if task_names:
            return task_names

    short_task = item.get("short_horizon_task")
    if isinstance(short_task, str) and short_task:
        return [short_task]
    return []


def _count_tasks_for_indices(
    episodes: list[dict],
    episode_indices: list[int],
    task_mapping: dict[int, str],
) -> Counter:
    indices = set(episode_indices)
    counter: Counter = Counter()
    for item in episodes:
        idx = item.get("episode_index")
        if idx not in indices:
            continue
        tasks = _episode_tasks(item, task_mapping)
        if tasks:
            for task in tasks:
                counter[task] += 1
        else:
            counter["<missing tasks>"] += 1
    return counter


def _report_task_ratio(
    train_counts: Counter,
    val_counts: Counter,
    train_fraction: float,
    *,
    tolerance: float,
    min_count: int,
) -> None:
    tasks = set(train_counts) | set(val_counts)
    if not tasks:
        print("[INFO] Task ratio check: no tasks found in episodes.jsonl")
        return

    print("[INFO] Task ratio check (train/val):")
    print("task\ttrain\tval\ttotal\ttrain_ratio\tdelta")
    for task in sorted(tasks, key=lambda t: (-(train_counts[t] + val_counts[t]), str(t))):
        train = train_counts[task]
        val = val_counts[task]
        total = train + val
        if total == 0:
            continue
        train_ratio = train / total
        delta = train_ratio - train_fraction
        if total < min_count:
            status = " (low-count)"
        elif abs(delta) > tolerance:
            status = " (out-of-range)"
        else:
            status = ""
        print(f"{task}\t{train}\t{val}\t{total}\t{train_ratio:.3f}\t{delta:+.3f}{status}")


def _sum_frames(episodes: list[dict]) -> int:
    total = 0
    for item in episodes:
        length = item.get("length")
        if isinstance(length, int):
            total += length
    return total


def _update_info(
    info: dict,
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    chunks_size: int,
    split_name: str,
) -> dict:
    updated = dict(info)
    updated["total_episodes"] = total_episodes
    updated["total_frames"] = total_frames
    updated["total_tasks"] = total_tasks
    updated["total_videos"] = total_videos
    updated["total_chunks"] = int(math.ceil(total_episodes / chunks_size)) if total_episodes > 0 else 0
    updated["splits"] = {split_name: f"0:{total_episodes}"}
    return updated


def _rewrite_parquet_episode_index(src_path: Path, dst_path: Path, new_index: int) -> None:
    table = pq.read_table(src_path, memory_map=True)
    if "episode_index" not in table.column_names:
        raise KeyError(f"episode_index column missing in {src_path}")
    field_idx = table.schema.get_field_index("episode_index")
    field = table.schema.field(field_idx)
    new_col = pa.array([new_index] * table.num_rows, type=field.type)
    updated = table.set_column(field_idx, "episode_index", new_col)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(updated, dst_path)


def _write_split_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    info: dict,
    episodes: list[dict],
    stats: list[dict],
    episode_indices: list[int],
    split_name: str,
    workers: int,
) -> None:
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"

    dst_root.mkdir(parents=True, exist_ok=True)
    dst_meta.mkdir(parents=True, exist_ok=True)

    for entry in src_meta.iterdir():
        if entry.name in {"episodes.jsonl", "episodes_stats.jsonl", "info.json"}:
            continue
        _copy_tree_entry(entry, dst_meta / entry.name)

    episodes_by_index = _episode_lookup(episodes)
    stats_by_index = _stats_lookup(stats)

    updated_episodes: list[dict] = []
    updated_stats: list[dict] = []

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(episode_indices)}

    for old_idx in episode_indices:
        new_idx = index_map[old_idx]
        episode_item = episodes_by_index.get(old_idx)
        if episode_item is None:
            raise KeyError(f"episode_index {old_idx} not found in episodes.jsonl")
        new_item = dict(episode_item)
        new_item["episode_index"] = new_idx
        new_item["original_episode_index"] = new_item.get("original_episode_index", old_idx)
        updated_episodes.append(new_item)

        stat_item = stats_by_index.get(old_idx)
        if stat_item is not None:
            new_stat = dict(stat_item)
            new_stat["episode_index"] = new_idx
            new_stat["original_episode_index"] = new_stat.get("original_episode_index", old_idx)
            updated_stats.append(new_stat)

    _write_jsonl(dst_meta / "episodes.jsonl", updated_episodes)
    if stats:
        _write_jsonl(dst_meta / "episodes_stats.jsonl", updated_stats)

    tasks_path = dst_meta / "tasks.jsonl"
    total_tasks = len(_read_jsonl(tasks_path)) if tasks_path.exists() else 0
    total_frames = _sum_frames(updated_episodes)

    video_keys = _get_video_keys(info)
    total_videos = len(video_keys) * len(updated_episodes)
    chunks_size = int(info.get("chunks_size") or 0)
    updated_info = _update_info(
        info,
        total_episodes=len(updated_episodes),
        total_frames=total_frames,
        total_tasks=total_tasks,
        total_videos=total_videos,
        chunks_size=chunks_size,
        split_name=split_name,
    )
    _write_json(dst_meta / "info.json", updated_info)

    data_template = info.get("data_path")
    if not data_template:
        raise KeyError("info.json missing data_path")
    video_template = info.get("video_path")

    def _process_episode(args: tuple[int, int]) -> None:
        new_idx, old_idx = args
        src_data = src_root / _format_template(
            data_template,
            episode_index=old_idx,
            chunk_size=chunks_size,
        )
        dst_data = dst_root / _format_template(
            data_template,
            episode_index=new_idx,
            chunk_size=chunks_size,
        )
        if not src_data.exists():
            raise FileNotFoundError(f"Missing data file: {src_data}")
        if new_idx == old_idx:
            _symlink_file(src_data.resolve(), dst_data)
        else:
            _rewrite_parquet_episode_index(src_data, dst_data, new_idx)

        if video_template:
            for key in video_keys:
                src_video = src_root / _format_template(
                    video_template,
                    episode_index=old_idx,
                    chunk_size=chunks_size,
                    video_key=key,
                )
                if not src_video.exists():
                    continue
                dst_video = dst_root / _format_template(
                    video_template,
                    episode_index=new_idx,
                    chunk_size=chunks_size,
                    video_key=key,
                )
                _symlink_file(src_video.resolve(), dst_video)

    tasks = [(index_map[old_idx], old_idx) for old_idx in episode_indices]
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_process_episode, tasks))
    else:
        for task in tasks:
            _process_episode(task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a LeRobot dataset into train/val datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--stratify-by-task",
        action="store_true",
        help="Split episodes per task group to preserve task ratio.",
    )
    parser.add_argument(
        "--stratify-task-mode",
        choices=("first", "all"),
        default="first",
        help="Use the first task label or all tasks (joined) for stratification.",
    )
    parser.add_argument(
        "--check-task-ratio",
        action="store_true",
        help="Report per-task train/val ratio based on episodes.jsonl.",
    )
    parser.add_argument(
        "--task-ratio-tolerance",
        type=float,
        default=0.05,
        help="Warn when per-task train ratio deviates beyond this tolerance.",
    )
    parser.add_argument(
        "--task-ratio-min-count",
        type=int,
        default=1,
        help="Minimum task count to apply ratio tolerance warnings.",
    )
    args = parser.parse_args()

    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1")

    if not _validate_outputs(args.train_output, args.val_output, overwrite=args.overwrite, skip_existing=args.skip_existing):
        return

    src_meta = args.dataset_root / "meta"
    info = _read_json(src_meta / "info.json")
    episodes = _load_episode_items(src_meta / "episodes.jsonl")
    stats = _read_jsonl(src_meta / "episodes_stats.jsonl")

    episode_indices = [int(item["episode_index"]) for item in episodes if "episode_index" in item]
    task_mapping = _load_task_mapping(src_meta / "tasks.jsonl")
    if args.stratify_by_task:
        train_indices, val_indices = _split_indices_stratified(
            episodes,
            args.train_fraction,
            args.seed,
            task_mapping,
            args.stratify_task_mode,
        )
        print(f"[INFO] Stratified split by task (mode={args.stratify_task_mode})")
    else:
        train_indices, val_indices = _split_indices(episode_indices, args.train_fraction, args.seed)

    print(f"[INFO] Episodes: total={len(episode_indices)} train={len(train_indices)} val={len(val_indices)}")

    workers = args.workers
    if workers <= 0:
        workers = max(1, min(8, os.cpu_count() or 1))

    _write_split_dataset(
        src_root=args.dataset_root,
        dst_root=args.train_output,
        info=info,
        episodes=episodes,
        stats=stats,
        episode_indices=train_indices,
        split_name="train",
        workers=workers,
    )
    _write_split_dataset(
        src_root=args.dataset_root,
        dst_root=args.val_output,
        info=info,
        episodes=episodes,
        stats=stats,
        episode_indices=val_indices,
        split_name="validation",
        workers=workers,
    )

    if args.check_task_ratio:
        train_counts = _count_tasks_for_indices(episodes, train_indices, task_mapping)
        val_counts = _count_tasks_for_indices(episodes, val_indices, task_mapping)
        _report_task_ratio(
            train_counts,
            val_counts,
            args.train_fraction,
            tolerance=args.task_ratio_tolerance,
            min_count=args.task_ratio_min_count,
        )

    print("[INFO] Split complete.")


if __name__ == "__main__":
    main()
