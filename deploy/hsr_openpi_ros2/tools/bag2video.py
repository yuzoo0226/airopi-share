#!/usr/bin/env python3
"""Render a ROS 2 bag of HSR camera topics into an mp4 for review.

The head and hand frames are placed side by side (head on the left), with a
caption strip carrying the episode index, the task string and the elapsed time,
so a reviewer can see what the robot saw and what it was asked to do.

Only ``rosbags`` (pure python), numpy and OpenCV are needed, so this runs both in
the ROS 2 container and in the training container.

    python bag2video.py --bag /home/bags/pick_001 --out pick_001.mp4 \
        --episodes 0,1,2 --fps 15
"""

from __future__ import annotations

import argparse
import bisect
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("bag2video")

DEFAULT_TOPICS = {
    "head": "/head_rgbd_sensor/rgb/image_rect_color/compressed",
    "hand": "/hand_camera/image_raw/compressed",
    "episode": "/hsr_pick_task/episode",
    "clock": "/clock",
}
FALLBACK_EPISODE_TOPICS = ["/hsr_pick_task/episode", "/hsr_random_motion/episode"]


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def read_streams(bag: pathlib.Path, topics: Dict[str, str]):
    from rosbags.highlevel import AnyReader  # noqa: PLC0415

    frames: Dict[str, List[Tuple[float, Any]]] = {"head": [], "hand": []}
    episodes: List[Tuple[float, str]] = []
    clock_wall: List[int] = []
    clock_sim: List[float] = []

    with AnyReader([bag]) as reader:
        available = {c.topic for c in reader.connections}
        ep_topic = next((t for t in FALLBACK_EPISODE_TOPICS if t in available), None)

        for conn, timestamp, raw in reader.messages(
            connections=[c for c in reader.connections if c.topic == topics["clock"]]
        ):
            msg = reader.deserialize(raw, conn.msgtype)
            clock_wall.append(timestamp)
            clock_sim.append(_stamp_to_sec(msg.clock))

        def wall_to_sim(ns: int) -> float:
            if len(clock_wall) < 2:
                return ns * 1e-9
            return float(
                np.interp(ns * 1e-9, np.asarray(clock_wall) * 1e-9, np.asarray(clock_sim))
            )

        wanted = {topics["head"]: "head", topics["hand"]: "hand"}
        if ep_topic:
            wanted[ep_topic] = "episode"
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, timestamp, raw in reader.messages(connections=conns):
            key = wanted[conn.topic]
            msg = reader.deserialize(raw, conn.msgtype)
            if key == "episode":
                episodes.append((wall_to_sim(timestamp), str(msg.data)))
                continue
            header = getattr(msg, "header", None)
            t = _stamp_to_sec(header.stamp) if header is not None else wall_to_sim(timestamp)
            frames[key].append((t, bytes(msg.data)))

    for key in frames:
        frames[key].sort(key=lambda kv: kv[0])
    episodes.sort(key=lambda kv: kv[0])
    return frames, episodes


def decode(payload: bytes) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    return img


def latest(stream: List[Tuple[float, Any]], t: float):
    if not stream:
        return None
    idx = bisect.bisect_right([s[0] for s in stream], t) - 1
    return None if idx < 0 else stream[idx][1]


def segments(episodes: List[Tuple[float, str]]) -> List[Tuple[int, float, float, str]]:
    out: List[Tuple[int, float, float, str]] = []
    open_ep: Optional[Tuple[int, float, str]] = None
    for t, data in episodes:
        parts = data.split(" ", 2)
        if parts[0] == "start" and len(parts) >= 2:
            task = parts[2] if len(parts) >= 3 else ""
            open_ep = (int(parts[1]), t, task)
        elif parts[0] == "end" and open_ep is not None:
            out.append((open_ep[0], open_ep[1], t, open_ep[2]))
            open_ep = None
    return out


def caption(width: int, lines: Sequence[str], height: int = 56) -> np.ndarray:
    strip = np.full((height, width, 3), 28, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            strip, line, (14, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA
        )
    return strip


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=480, help="Width of each camera pane.")
    ap.add_argument("--episodes", default="", help="Comma separated episode indices (default: all).")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = no limit.")
    ap.add_argument("--pad", type=float, default=0.5, help="Seconds kept around each episode.")
    ap.add_argument("--title", default="", help="Extra caption line.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    frames, episode_msgs = read_streams(args.bag, DEFAULT_TOPICS)
    logger.info("head=%d hand=%d episode msgs=%d", len(frames["head"]), len(frames["hand"]), len(episode_msgs))
    if not frames["head"] or not frames["hand"]:
        logger.error("no camera frames in the bag")
        return 1

    segs = segments(episode_msgs)
    if not segs:
        t0 = frames["head"][0][0]
        t1 = frames["head"][-1][0]
        segs = [(0, t0, t1, "")]
    wanted = {int(v) for v in args.episodes.split(",") if v.strip()} if args.episodes else None
    if wanted is not None:
        segs = [s for s in segs if s[0] in wanted]
    if args.max_episodes:
        segs = segs[: args.max_episodes]
    if not segs:
        logger.error("no episodes selected")
        return 1
    logger.info("rendering %d episode(s): %s", len(segs), [s[0] for s in segs])

    pane_w = args.width
    sample = decode(frames["head"][0][1])
    if sample is None:
        logger.error("could not decode the first head frame")
        return 1
    pane_h = int(round(pane_w * sample.shape[0] / sample.shape[1]))
    cap_h = 56
    size = (pane_w * 2, pane_h + cap_h)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    if not writer.isOpened():
        logger.error("could not open the video writer for %s", args.out)
        return 1

    dt = 1.0 / args.fps
    written = 0
    for index, start, end, task in segs:
        t = start - args.pad
        while t <= end + args.pad:
            head = decode(latest(frames["head"], t) or b"")
            hand = decode(latest(frames["hand"], t) or b"")
            if head is None or hand is None:
                t += dt
                continue
            left = cv2.resize(head, (pane_w, pane_h))
            right = cv2.resize(hand, (pane_w, pane_h))
            cv2.putText(left, "head", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(right, "hand", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            lines = [f"episode {index}   t={max(t - start, 0.0):5.1f}s", task or args.title]
            if args.title and task:
                lines[0] = f"{args.title}   |   {lines[0]}"
            frame = np.vstack([np.hstack([left, right]), caption(pane_w * 2, lines, cap_h)])
            writer.write(frame)
            written += 1
            t += dt
    writer.release()
    logger.info("wrote %s (%d frames, %.1f s)", args.out, written, written / args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
