#!/usr/bin/env python3
"""How often is the target object actually inside the head camera's view?

A policy can only learn where to drive from what it can see. The head camera is
the only sensor that sees the object before the gripper is nearly on top of it,
so if the object is off-frame for most of an episode the head image contributes
nothing and the base has to be steered from the hand camera alone -- which only
sees the object at the very end.

The scene is otherwise grey and tan, and every object in the library is a
saturated flat colour, so "pixels whose hue matches the object and whose
saturation is high" identifies it without depending on how Gazebo lit it --
matching raw RGB does not survive the shading:

    python head_view_coverage.py --bag /home/bags/gaze_fixed --bag /home/bags/gaze_object

Prints, per bag, the fraction of frames containing the object and the median
share of the image it covers.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np

HEAD_TOPICS = (
    "/head_rgbd_sensor/rgb/image_rect_color/compressed",
    "/head_rgbd_sensor/rgb/image_rect_color",
)
EPISODE_TOPICS = ("/hsr_pick_task/episode", "/pick_task/episode", "/hsr_openpi/episode")


def object_hues() -> Dict[str, Optional[float]]:
    """name -> OpenCV hue (0..179), or None for objects with no usable hue."""
    import cv2

    here = pathlib.Path(__file__).resolve().parents[1] / "hsr_openpi" / "hsr_openpi"
    sys.path.insert(0, str(here.parent))
    from hsr_openpi.gz_world import OBJECT_LIBRARY  # noqa: E402

    hues: Dict[str, Optional[float]] = {}
    for spec in OBJECT_LIBRARY:
        rgb = np.array([[list(spec.rgba[:3])]], dtype=np.float32)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0]
        # white_tube and anything else near grey has no hue to match on
        hues[spec.name] = None if hsv[1] < 0.15 else float(hsv[0]) / 2.0
    return hues


def decode(msg, topic: str) -> Optional[np.ndarray]:
    import cv2

    if topic.endswith("/compressed"):
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return None if image is None else image[:, :, ::-1]
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    image = data.reshape(msg.height, msg.width, -1)[:, :, :3]
    return image if msg.encoding == "rgb8" else image[:, :, ::-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", action="append", required=True, type=pathlib.Path)
    ap.add_argument("--hue-tolerance", type=float, default=12.0, help="OpenCV hue distance (0..179)")
    ap.add_argument("--min-saturation", type=int, default=70, help="Below this a pixel is scene, not object")
    ap.add_argument("--min-pixels", type=int, default=30, help="Pixels needed to call the object visible")
    args = ap.parse_args(argv)

    from rosbags.highlevel import AnyReader

    import cv2

    hues = object_hues()

    print(f"{'bag':<22s} {'frames':>7s} {'visible':>8s} {'median %':>9s} {'max %':>7s}")
    for bag in args.bag:
        with AnyReader([bag]) as reader:
            head_topic = next((t for t in HEAD_TOPICS if t in reader.topics), None)
            episode_topic = next((t for t in EPISODE_TOPICS if t in reader.topics), None)
            if head_topic is None:
                print(f"{bag.name:<22s} no head image topic")
                continue

            # the episode markers carry the task string, which names the object
            current = None
            marks: List[tuple] = []
            if episode_topic:
                for conn, stamp, raw in reader.messages(connections=[c for c in reader.connections if c.topic == episode_topic]):
                    text = reader.deserialize(raw, conn.msgtype).data
                    marks.append((stamp, text))

            def object_at(stamp: int) -> Optional[str]:
                name = None
                for mark_stamp, text in marks:
                    if mark_stamp > stamp:
                        break
                    if text.startswith("start "):
                        name = next((n for n in hues if n.replace("_", " ") in text), None)
                    elif text.startswith("end "):
                        name = None
                return name

            shares: List[float] = []
            visible = 0
            total = 0
            conns = [c for c in reader.connections if c.topic == head_topic]
            for conn, stamp, raw in reader.messages(connections=conns):
                name = object_at(stamp)
                if name is None or hues.get(name) is None:
                    continue
                image = decode(reader.deserialize(raw, conn.msgtype), head_topic)
                if image is None:
                    continue
                total += 1
                hsv = cv2.cvtColor(image[:, :, ::-1], cv2.COLOR_BGR2HSV)
                delta = np.abs(hsv[:, :, 0].astype(float) - hues[name])
                delta = np.minimum(delta, 180.0 - delta)  # hue wraps
                match = (delta < args.hue_tolerance) & (hsv[:, :, 1] >= args.min_saturation)
                count = int(match.sum())
                if count >= args.min_pixels:
                    visible += 1
                shares.append(100.0 * count / image.shape[0] / image.shape[1])

            if not total:
                print(f"{bag.name:<22s} no frames inside an episode with a hue to match")
                continue
            print(
                f"{bag.name:<22s} {total:7d} {100.0 * visible / total:7.1f}% "
                f"{np.median(shares):9.3f} {max(shares):7.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
