#!/usr/bin/env python3
"""Did the robot actually do what the recorded commands told it to?

The action labels in the dataset are the *commands* published on the controller
topics, not the motion that followed. Those are only the same thing while the
controller is accepting and tracking them. When it is not -- a trajectory
rejected for starting in the past, a joint against a limit, an arm pressed into
the table -- the bag still records a clean stream of commands next to a robot
that never moved, and training on it teaches the policy that its actions have no
effect. Nothing about the bag looks wrong.

So compare each arm/head command against the joint state a short while later:

    python check_command_tracking.py --bag /home/bags/pick_100

`residual` is |command - state after --horizon seconds|, in radians (metres for
arm_lift), reported per joint. A tracking controller drives this to near zero
between one command and the next; a rejected or blocked one leaves it at the
size of the motion that was asked for.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ARM_TOPIC = "/arm_trajectory_controller/joint_trajectory"
HEAD_TOPIC = "/head_trajectory_controller/joint_trajectory"
JOINT_TOPIC = "/joint_states"


class Series:
    """A joint's recorded positions, searchable by time.

    The stamps are extracted once: rebuilding them per lookup turns this into
    an O(commands x states) scan, which on a 90-episode bag never finishes.
    """

    def __init__(self, samples: List[Tuple[int, float]]) -> None:
        self.stamps = np.fromiter((s for s, _ in samples), dtype=np.int64, count=len(samples))
        self.values = np.fromiter((v for _, v in samples), dtype=float, count=len(samples))

    def at(self, stamp: int) -> Optional[float]:
        """Value at or just after `stamp`, or None past the end of the recording."""
        i = int(np.searchsorted(self.stamps, stamp))
        return None if i >= len(self.values) else float(self.values[i])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", action="append", required=True, type=pathlib.Path)
    ap.add_argument("--horizon", type=float, default=1.0, help="seconds after the command to look")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="residual above which a command counts as not followed",
    )
    ap.add_argument(
        "--windows",
        type=int,
        default=1,
        help="Split the recording into N equal spans and report each. A run that "
        "starts clean and degrades -- because something else on the machine got "
        "busy, or state accumulated in the simulator -- looks fine when the whole "
        "bag is averaged into one number.",
    )
    args = ap.parse_args(argv)

    from rosbags.highlevel import AnyReader

    horizon_ns = int(args.horizon * 1e9)

    for bag in args.bag:
        print(f"\n=== {bag.name} ===")
        with AnyReader([bag]) as reader:
            samples: Dict[str, List[Tuple[int, float]]] = {}
            conns = [c for c in reader.connections if c.topic == JOINT_TOPIC]
            for conn, stamp, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                for name, position in zip(msg.name, msg.position):
                    samples.setdefault(name, []).append((stamp, float(position)))
            if not samples:
                print("  no /joint_states in this bag")
                continue
            states = {name: Series(values) for name, values in samples.items()}

            residuals: Dict[str, List[Tuple[int, float]]] = {}
            for topic in (ARM_TOPIC, HEAD_TOPIC):
                conns = [c for c in reader.connections if c.topic == topic]
                if not conns:
                    continue
                for conn, stamp, raw in reader.messages(connections=conns):
                    msg = reader.deserialize(raw, conn.msgtype)
                    if not msg.points:
                        continue
                    target = msg.points[-1].positions
                    for name, want in zip(msg.joint_names, target):
                        series = states.get(name)
                        if series is None:
                            continue
                        got = series.at(stamp + horizon_ns)
                        if got is None:
                            continue
                        residuals.setdefault(name, []).append((stamp, abs(float(want) - got)))

            if not residuals:
                print("  no command topics in this bag")
                continue

            everything = [s for values in residuals.values() for s, _ in values]
            first, last = min(everything), max(everything)
            edges = np.linspace(first, last + 1, args.windows + 1)

            worst = 0.0
            for w in range(args.windows):
                lo, hi = edges[w], edges[w + 1]
                if args.windows > 1:
                    span = (lo - first) / 1e9, (hi - first) / 1e9
                    print(f"  -- {span[0]:.0f}s to {span[1]:.0f}s")
                print(f"  {'joint':<18s} {'commands':>9s} {'median':>8s} {'p90':>8s} {'not followed':>13s}")
                for name in sorted(residuals):
                    values = np.array([v for s, v in residuals[name] if lo <= s < hi])
                    if not len(values):
                        continue
                    share = 100.0 * float((values > args.tolerance).mean())
                    worst = max(worst, share)
                    print(
                        f"  {name:<18s} {len(values):9d} {np.median(values):8.4f} "
                        f"{np.percentile(values, 90):8.4f} {share:12.1f}%"
                    )
            verdict = "the controller tracked the commands" if worst < 20.0 else (
                "commands were NOT followed -- these action labels do not describe the motion"
            )
            print(f"  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
