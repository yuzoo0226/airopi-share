#!/usr/bin/env python3
"""Fix null action.rescale stats in episodes_stats.jsonl.

Default behavior removes the action.rescale key when its value is null.
Writes in-place and creates a .bak backup unless --no-backup is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def _process_line(line: str, mode: str) -> tuple[str, int]:
    line = line.strip()
    if not line:
        return "", 0
    obj = json.loads(line)
    stats = obj.get("stats")
    if isinstance(stats, dict) and "action.rescale" in stats and stats.get("action.rescale") is None:
        if mode == "remove-key":
            stats.pop("action.rescale", None)
        elif mode == "remove-episode":
            return "", 1
        return json.dumps(obj, ensure_ascii=False), 1
    return json.dumps(obj, ensure_ascii=False), 0


def fix_file(path: Path, *, mode: str, in_place: bool, backup: bool) -> tuple[int, int]:
    removed = 0
    total = 0

    if in_place:
        if backup and not path.with_suffix(path.suffix + ".bak").exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        tmp = Path(tmp_path)
        try:
            with path.open() as src, tmp.open("w") as dst:
                for line in src:
                    fixed, changed = _process_line(line, mode)
                    if fixed:
                        dst.write(fixed + "\n")
                    removed += changed
                    total += 1
            shutil.move(str(tmp), str(path))
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return total, removed

    with path.open() as src:
        for line in src:
            _, changed = _process_line(line, mode)
            removed += changed
            total += 1
    return total, removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-stats", required=True, help="Path to episodes_stats.jsonl")
    ap.add_argument(
        "--mode",
        default="remove-key",
        choices=("remove-key", "remove-episode"),
        help="How to handle null action.rescale entries",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only report counts, do not modify the file")
    ap.add_argument("--no-backup", action="store_true", help="Do not create a .bak backup on in-place edits")
    args = ap.parse_args()

    path = Path(args.episodes_stats)
    if not path.exists():
        raise FileNotFoundError(path)

    total, removed = fix_file(path, mode=args.mode, in_place=not args.dry_run, backup=not args.no_backup)

    print(f"[INFO] total_lines={total} changed={removed} mode={args.mode}")
    if args.dry_run:
        print("[INFO] dry-run: no changes written")
    else:
        print(f"[INFO] updated: {path}")


if __name__ == "__main__":
    main()
