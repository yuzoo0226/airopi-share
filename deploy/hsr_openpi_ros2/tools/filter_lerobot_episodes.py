#!/usr/bin/env python3
"""Build a LeRobot dataset with some episodes left out, without copying the data.

    filter_lerobot_episodes.py <src> <dst> --drop-failed
    filter_lerobot_episodes.py <src> <dst> --drop 9 198

A converted dataset usually contains episodes that should not train: a grasp that
missed, a teleoperator who restarted. Removing them by rewriting the tree costs a
full copy -- these datasets are mostly PNG frames inside parquet, tens of
gigabytes -- and rewriting the parquet contents costs it twice, because the
episode_index and index columns inside each file would have to be renumbered to
stay consistent with the metadata.

Neither is necessary. The kept episodes keep their original numbering, so the
parquet contents are untouched and can be hardlinked; what changes is only the
three metadata files that say which episodes exist. The result has gaps in its
episode numbering, and that is supported: FastLeRobotDataset builds
`{episode_index: position}` from the episode list it is given rather than
assuming a contiguous range (fast_lerobot_dataset.py, _refresh_episode_index_cache).

Hardlinks mean src and dst must be on one filesystem, and that editing a parquet
in one tree would change it in the other. Nothing in this pipeline writes to a
converted dataset, and --copy is there for when that stops being true.

Note that `--drop-failed` reads `task_success`, which is what the collection
interface recorded about the *episode*. It is not a data-integrity check: an
episode can be marked successful and still hold joint values the robot cannot
reach. rosbag2_to_lerobot.py checks the ranges at conversion time; if the dataset
came from somewhere else, check them before trusting it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from typing import Iterable


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("dst", type=pathlib.Path)
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="episode_index values to leave out")
    ap.add_argument(
        "--drop-failed",
        action="store_true",
        help="also drop every episode whose metadata says task_success is false",
    )
    ap.add_argument(
        "--drop-short-horizon-failed",
        action="store_true",
        help="also drop success_short_horizon_task=false, i.e. episodes that are "
        "fine in themselves but belong to a sequence that did not finish",
    )
    ap.add_argument("--copy", action="store_true", help="copy the parquet files instead of hardlinking")
    args = ap.parse_args(argv)

    meta = args.src / "meta"
    episodes = _read_jsonl(meta / "episodes.jsonl")

    drop = set(args.drop)
    if args.drop_failed:
        drop |= {e["episode_index"] for e in episodes if not e.get("task_success", True)}
    if args.drop_short_horizon_failed:
        drop |= {e["episode_index"] for e in episodes if not e.get("success_short_horizon_task", True)}

    kept = [e for e in episodes if e["episode_index"] not in drop]
    if not kept:
        print("[ERROR] every episode was dropped", file=sys.stderr)
        return 1
    missing = drop - {e["episode_index"] for e in episodes}
    if missing:
        print(f"[WARN] not in the dataset, ignored: {sorted(missing)}", file=sys.stderr)

    (args.dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (args.dst / "meta").mkdir(parents=True, exist_ok=True)

    _write_jsonl(args.dst / "meta" / "episodes.jsonl", kept)
    keep_idx = {e["episode_index"] for e in kept}
    _write_jsonl(
        args.dst / "meta" / "episodes_stats.jsonl",
        (s for s in _read_jsonl(meta / "episodes_stats.jsonl") if s["episode_index"] in keep_idx),
    )
    shutil.copy(meta / "tasks.jsonl", args.dst / "meta" / "tasks.jsonl")

    frames = sum(e["length"] for e in kept)
    info = json.loads((meta / "info.json").read_text())
    info["total_episodes"] = len(kept)
    info["total_frames"] = frames
    # Nothing in the training path reads splits, but leaving it at the old count
    # would make the file describe a dataset that no longer exists.
    info["splits"] = {"train": f"0:{len(kept)}"}
    (args.dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    for e in kept:
        name = f"episode_{e['episode_index']:06d}.parquet"
        source = args.src / "data" / "chunk-000" / name
        target = args.dst / "data" / "chunk-000" / name
        if not source.is_file():
            print(f"[ERROR] {source} is in the metadata but not on disk", file=sys.stderr)
            return 1
        if target.exists():
            target.unlink()
        if args.copy:
            shutil.copy(source, target)
        else:
            target.hardlink_to(source)

    dropped = sorted(drop & {e["episode_index"] for e in episodes})
    print(f"kept {len(kept)} episodes / {frames} frames")
    print(f"dropped {len(dropped)}: {dropped}")
    print(f"wrote {args.dst}")
    print("\nnorm_stats still has to be computed from the filtered tree:")
    print(
        f"  scripts/aggregate_stats_fast.py --episodes-stats {args.dst}/meta/episodes_stats.jsonl \\\n"
        f"      --chunk-dir {args.dst}/data --output-file <assets>/norm_stats.json \\\n"
        f"      --action-column action.relative --action-mode relative --min-std 0.01"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
