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


def _sample(series: List[Tuple[int, float]], stamp: int) -> Optional[float]:
    """Value of a (stamp, value) series at or just after `stamp`."""
    stamps = [s for s, _ in series]
    i = int(np.searchsorted(stamps, stamp))
    if i >= len(series):
        return None
    return series[i][1]


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
    args = ap.parse_args(argv)

    from rosbags.highlevel import AnyReader

    horizon_ns = int(args.horizon * 1e9)

    for bag in args.bag:
        print(f"\n=== {bag.name} ===")
        with AnyReader([bag]) as reader:
            states: Dict[str, List[Tuple[int, float]]] = {}
            conns = [c for c in reader.connections if c.topic == JOINT_TOPIC]
            for conn, stamp, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                for name, position in zip(msg.name, msg.position):
                    states.setdefault(name, []).append((stamp, float(position)))
            if not states:
                print("  no /joint_states in this bag")
                continue

            residuals: Dict[str, List[float]] = {}
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
                        if not series:
                            continue
                        got = _sample(series, stamp + horizon_ns)
                        if got is None:
                            continue
                        residuals.setdefault(name, []).append(abs(float(want) - got))

            if not residuals:
                print("  no command topics in this bag")
                continue

            print(f"  {'joint':<18s} {'commands':>9s} {'median':>8s} {'p90':>8s} {'not followed':>13s}")
            worst = 0.0
            for name in sorted(residuals):
                values = np.array(residuals[name])
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
