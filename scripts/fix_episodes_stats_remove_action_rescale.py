#!/usr/bin/env python3
"""Remove action.rescale entries from LeRobot episodes_stats.jsonl.

Writes a new file to avoid in-place corruption. Use --inplace to overwrite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jsonlines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-stats", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    src = args.episodes_stats
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    def next_backup_path(path: Path) -> Path:
        base = path.with_suffix(path.suffix + ".bak")
        if not base.exists():
            return base
        i = 1
        while True:
            cand = path.with_suffix(path.suffix + f".bak{i}")
            if not cand.exists():
                return cand
            i += 1

    if args.output:
        dst = args.output
        inplace = False
    else:
        inplace = True
        dst = src.with_suffix(src.suffix + ".tmp")

    removed = 0
    total = 0

    with jsonlines.open(src, "r") as reader, jsonlines.open(dst, "w") as writer:
        for item in reader:
            total += 1
            stats = item.get("stats")
            if isinstance(stats, dict) and "action.rescale" in stats:
                stats.pop("action.rescale", None)
                removed += 1
            writer.write(item)

    if inplace:
        if not args.no_backup:
            backup = next_backup_path(src)
            src.replace(backup)
        dst.replace(src)

    print(f"processed={total} removed_action_rescale={removed} output={src if inplace else dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
