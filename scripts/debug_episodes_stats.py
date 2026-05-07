#!/usr/bin/env python3
"""Scan LeRobot episodes_stats.jsonl for malformed entries.

Finds episodes where any feature stats are not a dict with keys
[min, max, mean, std, count]. Also verifies leaf types after
conversion to numpy arrays.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jsonlines
import numpy as np

REQUIRED_KEYS = {"min", "max", "mean", "std", "count"}


def iter_jsonlines(path: Path):
    with jsonlines.open(path, "r") as reader:
        for item in reader:
            yield item


def is_stats_dict(value) -> bool:
    return isinstance(value, dict) and REQUIRED_KEYS.issubset(value.keys())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-stats", required=True, type=Path)
    ap.add_argument("--max-issues", type=int, default=20)
    args = ap.parse_args()

    bad_entries = 0
    total = 0

    for item in iter_jsonlines(args.episodes_stats):
        total += 1
        ep_index = item.get("episode_index")
        stats = item.get("stats")

        if not isinstance(stats, dict):
            bad_entries += 1
            print(f"[BAD] episode_index={ep_index}: stats is {type(stats)}", file=sys.stderr)
            if bad_entries >= args.max_issues:
                break
            continue

        for fkey, fval in stats.items():
            if not is_stats_dict(fval):
                bad_entries += 1
                print(
                    f"[BAD] episode_index={ep_index} feature={fkey}: expected dict with {sorted(REQUIRED_KEYS)}, got {type(fval)}",
                    file=sys.stderr,
                )
                # Print a short preview for debugging
                preview = str(fval)
                print(f"  preview={preview[:200]}", file=sys.stderr)
                if bad_entries >= args.max_issues:
                    break
        if bad_entries >= args.max_issues:
            break

    # Second pass: check that dict leaves can be converted to numpy arrays
    if bad_entries == 0:
        for item in iter_jsonlines(args.episodes_stats):
            ep_index = item.get("episode_index")
            stats = item.get("stats", {})
            for fkey, fval in stats.items():
                if not is_stats_dict(fval):
                    continue
                for k, v in fval.items():
                    try:
                        arr = np.array(v)
                    except Exception as exc:
                        bad_entries += 1
                        print(
                            f"[BAD] episode_index={ep_index} feature={fkey} key={k}: cannot convert to numpy array: {exc}",
                            file=sys.stderr,
                        )
                        if bad_entries >= args.max_issues:
                            break
                    else:
                        if arr.ndim == 0:
                            bad_entries += 1
                            print(
                                f"[BAD] episode_index={ep_index} feature={fkey} key={k}: array has ndim=0",
                                file=sys.stderr,
                            )
                            if bad_entries >= args.max_issues:
                                break
                if bad_entries >= args.max_issues:
                    break
            if bad_entries >= args.max_issues:
                break

    if bad_entries == 0:
        print(f"OK: scanned {total} episodes, no malformed stats found")
        return 0

    print(f"Found {bad_entries} issues while scanning {total} episodes", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
