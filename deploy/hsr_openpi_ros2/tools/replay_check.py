#!/usr/bin/env python3
"""Open-loop check: does the policy reproduce the demonstrations it was trained on?

Frames are read straight out of the LeRobot dataset, sent to the running policy
server exactly as the ROS 2 node would send them, and the first action of the
returned chunk is compared against the action recorded for that frame.

This separates the two ways a trained policy can disappoint:

* **large open-loop error** — the problem is upstream: the dataset, the
  normalisation stats, the image channel order, or simply not enough training;
* **small open-loop error but poor closed-loop success** — the policy has
  learned the mapping and the problem is compounding error / distribution shift
  during execution, which is a control question, not a data question.

    python replay_check.py --dataset /home/datasets/lerobot_datasets/hsr_gazebo_pick \
        --host 127.0.0.1 --port 8010 --episodes 0,1,2 --stride 5
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

STATE_NAMES = [
    "arm_lift", "arm_flex", "arm_roll", "wrist_flex", "wrist_roll",
    "hand_motor", "head_pan", "head_tilt",
]
ACTION_NAMES = STATE_NAMES[:5] + ["gripper", "head_pan", "head_tilt", "base_x", "base_y", "base_t"]


# --- msgpack wire format, identical to openpi_client.msgpack_numpy ----------- #
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class PolicyClient:
    def __init__(self, host: str, port: int):
        import msgpack
        import websockets.sync.client

        self._pack = functools.partial(msgpack.packb, default=_pack_array)
        self._unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
        self._ws = websockets.sync.client.connect(f"ws://{host}:{port}", compression=None, max_size=None)
        self.metadata = self._unpack(self._ws.recv())

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._ws.send(self._pack(obs))
        reply = self._ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"policy server error:\n{reply}")
        return self._unpack(reply)


def decode_png(payload: Dict[str, Any]) -> np.ndarray:
    """Dataset images are stored with the channel order they were converted in."""
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(payload["bytes"])).convert("RGB"))


def _dataset_action_std(dataset: pathlib.Path) -> np.ndarray:
    """Per-dimension standard deviation of action.relative over the whole dataset."""
    import pyarrow.parquet as pq

    chunks = []
    for path in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["action.relative"])
        chunks.append(np.stack(table["action.relative"].to_numpy(zero_copy_only=False)))
    if not chunks:
        return np.ones(len(ACTION_NAMES), dtype=np.float32)
    return np.concatenate(chunks).std(axis=0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=pathlib.Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--episodes", default="0,1,2", help="Comma separated episode indices.")
    ap.add_argument("--stride", type=int, default=5, help="Use every Nth frame.")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Write the per-frame comparison as json.")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq

    tasks = {
        json.loads(line)["task_index"]: json.loads(line)["task"]
        for line in (args.dataset / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    }

    wanted = [int(v) for v in args.episodes.split(",") if v.strip()]
    files = []
    for ep in wanted:
        matches = list((args.dataset / "data").rglob(f"episode_{ep:06d}.parquet"))
        if not matches:
            print(f"[WARN] episode {ep} not found", file=sys.stderr)
            continue
        files.append((ep, matches[0]))
    if not files:
        print("[ERROR] no episodes found", file=sys.stderr)
        return 1

    client = PolicyClient(args.host, args.port)
    print(f"connected: {client.metadata.get('config_name')} @ {client.metadata.get('checkpoint_dir')}")

    rows: List[dict] = []
    for ep, path in files:
        table = pq.read_table(path)
        n = table.num_rows
        for i in range(0, n, args.stride):
            if len(rows) >= args.max_frames:
                break
            state = np.asarray(table["observation.state"][i].as_py(), dtype=np.float32)
            target = np.asarray(table["action.relative"][i].as_py(), dtype=np.float32)
            task = tasks.get(table["task_index"][i].as_py(), "")
            obs = {
                "head_rgb": decode_png(table["observation.image.head"][i].as_py()),
                "hand_rgb": decode_png(table["observation.image.hand"][i].as_py()),
                "state": state,
                "prompt": task,
            }
            pred = np.asarray(client.infer(obs)["actions"], dtype=np.float32)[0]
            rows.append(
                {
                    "episode": ep,
                    "frame": i,
                    "task": task,
                    "target": target.tolist(),
                    "pred": pred[: len(target)].tolist(),
                }
            )
        if len(rows) >= args.max_frames:
            break

    T = np.array([r["target"] for r in rows], dtype=np.float32)
    P = np.array([r["pred"] for r in rows], dtype=np.float32)
    err = np.abs(P - T)

    # Normalise against the spread of the *whole* dataset, not of the sampled
    # frames: a handful of frames from one episode has almost no variance, which
    # would make every ratio meaningless.
    dataset_std = _dataset_action_std(args.dataset)

    print(f"\n{len(rows)} frames from episodes {[e for e, _ in files]}")
    print(f"{'dim':12s} {'dataset std':>12s} {'MAE':>9s} {'MAE/std':>9s}")
    for i, name in enumerate(ACTION_NAMES):
        std = float(dataset_std[i])
        mae = float(err[:, i].mean())
        ratio = mae / std if std > 1e-4 else float("nan")
        print(f"{name:12s} {std:12.4f} {mae:9.4f} {ratio:9.2f}")

    moving = [i for i, _ in enumerate(ACTION_NAMES) if dataset_std[i] > 1e-4]
    print(
        f"\nover the {len(moving)} dimensions that actually move "
        f"({', '.join(ACTION_NAMES[i] for i in moving)}):"
    )
    print(f"  mean |error| = {err[:, moving].mean():.4f}")
    print(
        "  normalised   = "
        f"{np.mean([err[:, i].mean() / dataset_std[i] for i in moving]):.2f} x dataset std"
    )
    print("\n  < 0.3 x std  : the policy reproduces the demonstrations well")
    print("  ~ 1.0 x std  : no better than predicting the dataset mean")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"frames": rows}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
