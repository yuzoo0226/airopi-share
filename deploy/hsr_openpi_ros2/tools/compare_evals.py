#!/usr/bin/env python3
"""Put two evaluation runs side by side, on numbers fixed in advance.

    python compare_evals.py eval_a.json eval_b.json

Success rate over twenty episodes is a coarse number: one episode is five
points, so a run can look better or worse than another by pure luck. These
columns each measure a different part of the task, so a change in one of them
says something the success rate cannot.

`gripper never closed` is the sharpest of them. hand_motor is 1.2 fully open and
around 0.5-0.8 when the fingers are around an object; a failure that ends near
1.0 did not attempt a grasp at all, which is a different problem from reaching
for the object and missing. In the 10k run six of thirteen failures were of that
kind.

Deciding what to compare before seeing the second run is the point. This tool
exists so the same arithmetic runs on both, rather than a fresh reading of
whichever numbers happen to support the story afterwards.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any, Dict, List

TOLERANCE = 0.045          # grasp_xy_tolerance in pick_task
GRIPPER_OPEN_ENOUGH = 0.87  # above this the fingers never closed on anything


def summarise(path: pathlib.Path) -> Dict[str, Any]:
    d = json.loads(path.read_text())
    results: List[dict] = d["results"]
    ok = [r for r in results if r["success"]]
    bad = [r for r in results if not r["success"]]

    def dxy(r: dict) -> float | None:
        g = r.get("grasp_condition")
        return g.get("dxy") if isinstance(g, dict) else None

    def motor(r: dict) -> float | None:
        g = r.get("grasp_condition")
        return g.get("hand_motor") if isinstance(g, dict) else None

    bad_dxy = sorted(v for v in (dxy(r) for r in bad) if v is not None)
    never_closed = [r for r in bad if (motor(r) or 0) >= GRIPPER_OPEN_ENOUGH]
    closed_missed = [r for r in bad if (motor(r) or 0) < GRIPPER_OPEN_ENOUGH]
    lifted = [r for r in bad if (r.get("final_z") or 0) - (r.get("spawn_z") or 0) > 0.05]

    return {
        "name": path.stem,
        "episodes": len(results),
        "success": len(ok),
        "rate": 100.0 * len(ok) / max(len(results), 1),
        "never_closed": len(never_closed),
        "closed_but_missed": len(closed_missed),
        "lifted_but_failed": len(lifted),
        "dxy_median": statistics.median(bad_dxy) if bad_dxy else float("nan"),
        "dxy_within_1cm": sum(1 for v in bad_dxy if 0 < v - TOLERANCE <= 0.01),
        "dxy_over_half_metre": sum(1 for v in bad_dxy if v > 0.5),
    }


ROWS = [
    ("episodes", "episodes", "{:d}"),
    ("success", "successes", "{:d}"),
    ("rate", "success rate %", "{:.1f}"),
    ("never_closed", "gripper never closed", "{:d}"),
    ("closed_but_missed", "closed but missed", "{:d}"),
    ("lifted_but_failed", "lifted yet scored a failure", "{:d}"),
    ("dxy_median", "distance to object, median m", "{:.4f}"),
    ("dxy_within_1cm", "failures within 1 cm of tolerance", "{:d}"),
    ("dxy_over_half_metre", "failures over 0.5 m away", "{:d}"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evals", nargs="+", type=pathlib.Path)
    args = ap.parse_args(argv)

    cols = [summarise(p) for p in args.evals]
    width = max(len(c["name"]) for c in cols) + 2
    print(f"{'':<34s}" + "".join(f"{c['name']:>{width}s}" for c in cols))
    for key, label, fmt in ROWS:
        print(f"{label:<34s}" + "".join(f"{fmt.format(c[key]):>{width}s}" for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
