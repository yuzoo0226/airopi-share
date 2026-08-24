#!/usr/bin/env python3
"""Convert a ROS 2 bag of HSR teleop / random motion into a LeRobot dataset.

The output matches what ``LeRobotHSRDataConfig`` (``action_mode: relative``)
reads during training:

======================== ============== ==========================================
column                   dtype/shape    meaning
======================== ============== ==========================================
observation.image.head   image (H,W,3)  head RGBD sensor colour frame
observation.image.hand   image (H,W,3)  hand camera frame
observation.state        float32 (8,)   measured joint positions, in the order
                                        arm_lift, arm_flex, arm_roll, wrist_flex,
                                        wrist_roll, hand_motor, head_pan, head_tilt
action.relative          float32 (11,)  arm(5) and head(2) as **deltas** from the
                                        measured state, gripper(1) absolute,
                                        base(3) as vx, vy, wz velocities
======================== ============== ==========================================

`action.relative` is exactly the inverse of what the deployment node applies::

    command = action + [state[:5], 0, state[6:8], 0, 0, 0]

so a policy trained on this data can be executed by ``hsr_openpi_node`` unchanged.

The bag is read with the pure-python ``rosbags`` library, so this script does not
need a ROS installation — run it inside the training container, which is the one
that has LeRobot:

    pip install rosbags
    python deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py \
        --bag /home/openpi/_bags/random_01 \
        --repo-id lerobot_datasets/hsr_gazebo_random \
        --root /home/datasets \
        --fps 10

Timestamps
----------
rosbag2 stamps every message with the recorder's *wall* clock while the
simulation runs on ``/clock``. Messages that carry a header (JointState,
CompressedImage, JointTrajectory) already hold simulation time; for the ones that
do not (Twist, String) the recorded ``/clock`` stream is used to map wall time to
simulation time.
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import logging
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("rosbag2_to_lerobot")

STATE_NAMES = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "hand_motor_joint",
    "head_pan_joint",
    "head_tilt_joint",
]
ACTION_NAMES = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "hand_motor_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "base_x",
    "base_y",
    "base_theta",
]
ARM_JOINTS = STATE_NAMES[:5]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]

DEFAULT_TOPICS = {
    "joint_states": "/joint_states",
    "head_image": "/head_rgbd_sensor/rgb/image_rect_color/compressed",
    "hand_image": "/hand_camera/image_raw/compressed",
    "arm_command": "/arm_trajectory_controller/joint_trajectory",
    "head_command": "/head_trajectory_controller/joint_trajectory",
    "gripper_command": "/gripper_controller/joint_trajectory",
    "base_command": "/omni_base_controller/cmd_vel",
    "control_mode": "/control_mode",
    "episode": "/hsr_random_motion/episode",
    "clock": "/clock",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class WallToSimClock:
    """Maps rosbag2 wall-clock receive times onto the /clock timeline."""

    def __init__(self, wall_ns: Sequence[int], sim_s: Sequence[float]):
        self.wall = np.asarray(wall_ns, dtype=np.float64) * 1e-9
        self.sim = np.asarray(sim_s, dtype=np.float64)
        self.valid = self.wall.size >= 2

    def __call__(self, wall_ns: int) -> float:
        t = wall_ns * 1e-9
        if not self.valid:
            return t
        return float(np.interp(t, self.wall, self.sim))


@dataclasses.dataclass
class Series:
    """Time-sorted samples with zero-order-hold lookup."""

    times: List[float] = dataclasses.field(default_factory=list)
    values: List[Any] = dataclasses.field(default_factory=list)

    def add(self, t: float, v: Any) -> None:
        self.times.append(t)
        self.values.append(v)

    def finish(self) -> None:
        if not self.times:
            return
        order = np.argsort(np.asarray(self.times))
        self.times = [self.times[i] for i in order]
        self.values = [self.values[i] for i in order]

    def at(self, t: float) -> Tuple[Optional[Any], float]:
        """Latest value at or before ``t`` and its age in seconds."""
        if not self.times:
            return None, float("inf")
        idx = bisect.bisect_right(self.times, t) - 1
        if idx < 0:
            return None, float("inf")
        return self.values[idx], t - self.times[idx]

    def __len__(self) -> int:
        return len(self.times)


# --------------------------------------------------------------------------- #
# bag reading
# --------------------------------------------------------------------------- #
def read_bag(bag_path: pathlib.Path, topics: Dict[str, str]) -> Dict[str, Series]:
    from rosbags.highlevel import AnyReader  # noqa: PLC0415

    wanted = {v: k for k, v in topics.items()}
    series: Dict[str, Series] = {k: Series() for k in topics}
    clock_wall: List[int] = []
    clock_sim: List[float] = []

    with AnyReader([bag_path]) as reader:
        available = {c.topic for c in reader.connections}
        missing = [t for t in topics.values() if t not in available]
        if missing:
            logger.warning("topics missing from the bag: %s", ", ".join(missing))

        # First pass: /clock, so header-less messages can be placed on sim time.
        clock_conns = [c for c in reader.connections if c.topic == topics["clock"]]
        for conn, timestamp, raw in reader.messages(connections=clock_conns):
            msg = reader.deserialize(raw, conn.msgtype)
            clock_wall.append(timestamp)
            clock_sim.append(_stamp_to_sec(msg.clock))
        clock = WallToSimClock(clock_wall, clock_sim)
        if not clock.valid:
            logger.warning("no /clock in the bag; falling back to the recorder wall clock")

        conns = [c for c in reader.connections if c.topic in wanted and c.topic != topics["clock"]]
        for conn, timestamp, raw in reader.messages(connections=conns):
            key = wanted[conn.topic]
            msg = reader.deserialize(raw, conn.msgtype)
            header = getattr(msg, "header", None)
            if header is not None and (header.stamp.sec or header.stamp.nanosec):
                t = _stamp_to_sec(header.stamp)
            else:
                t = clock(timestamp)
            series[key].add(t, msg)

    logger.info("%-16s %-58s %6d msgs", "clock", topics["clock"], len(clock_wall))
    for key, s in series.items():
        if key == "clock":
            continue  # consumed above to build the wall -> sim time mapping
        s.finish()
        logger.info("%-16s %-58s %6d msgs", key, topics[key], len(s))
    return series


# --------------------------------------------------------------------------- #
# episode segmentation
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Episode:
    index: int
    start: float
    end: float
    task: str


def segment_episodes(series: Dict[str, Series], default_task: str) -> List[Episode]:
    """Prefer the explicit episode topic; fall back to /control_mode == auto."""
    episodes: List[Episode] = []
    ep = series.get("episode")
    if ep is not None and len(ep) > 0:
        open_ep: Optional[Tuple[int, float, str]] = None
        for t, msg in zip(ep.times, ep.values):
            parts = str(msg.data).split(" ", 2)
            kind = parts[0]
            if kind == "start" and len(parts) >= 2:
                task = parts[2] if len(parts) >= 3 and parts[2] else default_task
                open_ep = (int(parts[1]), t, task)
            elif kind == "end" and open_ep is not None:
                episodes.append(Episode(open_ep[0], open_ep[1], t, open_ep[2]))
                open_ep = None
        if episodes:
            return episodes
        logger.warning("episode topic present but no start/end pairs; falling back to /control_mode")

    cm = series.get("control_mode")
    if cm is None or len(cm) == 0:
        raise RuntimeError("neither the episode topic nor /control_mode is usable for segmentation")

    start: Optional[float] = None
    idx = 0
    for t, msg in zip(cm.times, cm.values):
        active = str(msg.data) == "auto"
        if active and start is None:
            start = t
        elif not active and start is not None:
            episodes.append(Episode(idx, start, t, default_task))
            idx += 1
            start = None
    if start is not None:
        episodes.append(Episode(idx, start, cm.times[-1], default_task))
    return episodes


# --------------------------------------------------------------------------- #
# frame assembly
# --------------------------------------------------------------------------- #
def joint_positions(msg, names: Sequence[str]) -> Optional[np.ndarray]:
    index = {n: i for i, n in enumerate(msg.name)}
    try:
        return np.asarray([msg.position[index[n]] for n in names], dtype=np.float32)
    except KeyError:
        return None


def trajectory_positions(msg, names: Sequence[str]) -> Optional[np.ndarray]:
    if not msg.points:
        return None
    index = {n: i for i, n in enumerate(msg.joint_names)}
    point = msg.points[0]
    try:
        return np.asarray([point.positions[index[n]] for n in names], dtype=np.float32)
    except (KeyError, IndexError):
        return None


def decode_compressed(msg, image_order: str) -> Optional[np.ndarray]:
    import cv2  # noqa: PLC0415

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return None
    return img[:, :, ::-1] if image_order == "rgb" else img


def build_frames(
    series: Dict[str, Series],
    episode: Episode,
    *,
    fps: float,
    image_order: str,
    max_age_s: float,
) -> List[Dict[str, Any]]:
    dt = 1.0 / fps
    times = np.arange(episode.start, episode.end, dt)
    frames: List[Dict[str, Any]] = []
    dropped = {"state": 0, "image": 0, "command": 0}

    for t in times:
        js_msg, js_age = series["joint_states"].at(t)
        if js_msg is None or js_age > max_age_s:
            dropped["state"] += 1
            continue
        state = joint_positions(js_msg, STATE_NAMES)
        if state is None:
            dropped["state"] += 1
            continue

        head_msg, head_age = series["head_image"].at(t)
        hand_msg, hand_age = series["hand_image"].at(t)
        if head_msg is None or hand_msg is None or max(head_age, hand_age) > max_age_s:
            dropped["image"] += 1
            continue
        head_img = decode_compressed(head_msg, image_order)
        hand_img = decode_compressed(hand_msg, image_order)
        if head_img is None or hand_img is None:
            dropped["image"] += 1
            continue

        arm_msg, arm_age = series["arm_command"].at(t)
        head_cmd_msg, head_cmd_age = series["head_command"].at(t)
        grip_msg, grip_age = series["gripper_command"].at(t)
        base_msg, base_age = series["base_command"].at(t)
        if arm_msg is None or head_cmd_msg is None or grip_msg is None:
            dropped["command"] += 1
            continue
        if max(arm_age, head_cmd_age, grip_age) > max_age_s:
            dropped["command"] += 1
            continue

        arm_cmd = trajectory_positions(arm_msg, ARM_JOINTS)
        head_cmd = trajectory_positions(head_cmd_msg, HEAD_JOINTS)
        grip_cmd = trajectory_positions(grip_msg, ["hand_motor_joint"])
        if arm_cmd is None or head_cmd is None or grip_cmd is None:
            dropped["command"] += 1
            continue

        if base_msg is None or base_age > max_age_s:
            base = np.zeros(3, dtype=np.float32)
        else:
            base = np.asarray(
                [base_msg.linear.x, base_msg.linear.y, base_msg.angular.z], dtype=np.float32
            )

        action = np.zeros(11, dtype=np.float32)
        action[0:5] = arm_cmd - state[0:5]     # relative arm target
        action[5] = grip_cmd[0]                # absolute gripper opening
        action[6:8] = head_cmd - state[6:8]    # relative head target
        action[8:11] = base                    # base velocity

        frames.append(
            {
                "observation.image.head": head_img,
                "observation.image.hand": hand_img,
                "observation.state": state,
                "action.relative": action,
                # This LeRobot revision takes the natural language task per frame
                # (save_episode() has no `task` argument); it is turned into a
                # task index when the episode is saved, and openpi's
                # PromptFromLeRobotTask turns it back into the prompt.
                "task": episode.task,
            }
        )

    if any(dropped.values()):
        logger.info(
            "episode %d: dropped %d/%d frames (state=%d image=%d command=%d)",
            episode.index,
            sum(dropped.values()),
            len(times),
            dropped["state"],
            dropped["image"],
            dropped["command"],
        )
    return frames


# --------------------------------------------------------------------------- #
# LeRobot writing
# --------------------------------------------------------------------------- #
def write_lerobot(
    episodes_frames: List[Tuple[Episode, List[Dict[str, Any]]]],
    *,
    repo_id: str,
    root: pathlib.Path,
    fps: float,
    overwrite: bool,
    robot_type: str,
) -> pathlib.Path:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    out = root / repo_id
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} already exists (pass --overwrite)")
        shutil.rmtree(out)

    head_shape = episodes_frames[0][1][0]["observation.image.head"].shape
    hand_shape = episodes_frames[0][1][0]["observation.image.hand"].shape

    features = {
        "observation.image.head": {
            "dtype": "image",
            "shape": tuple(head_shape),
            "names": ["height", "width", "channel"],
        },
        "observation.image.hand": {
            "dtype": "image",
            "shape": tuple(hand_shape),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action.relative": {"dtype": "float32", "shape": (11,), "names": ACTION_NAMES},
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=out,
        robot_type=robot_type,
        fps=int(round(fps)),
        features=features,
        use_videos=False,
        image_writer_threads=8,
        image_writer_processes=2,
    )

    for episode, frames in episodes_frames:
        for frame in frames:
            dataset.add_frame(dict(frame))
        dataset.save_episode()
        logger.info("wrote episode %d (%d frames, task=%r)", episode.index, len(frames), episode.task)

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    return out


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True, type=pathlib.Path, help="rosbag2 directory (mcap or sqlite3).")
    p.add_argument("--repo-id", required=True, help="LeRobot repo id, e.g. lerobot_datasets/hsr_gazebo_random.")
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/home/datasets"), help="Dataset root.")
    p.add_argument("--fps", type=float, default=10.0, help="Resampling rate of the dataset.")
    p.add_argument(
        "--image-order",
        default="bgr",
        choices=["bgr", "rgb"],
        help="Channel order stored in the dataset. Must match the deployment "
        "parameter policy_image_order (default: bgr on both sides).",
    )
    p.add_argument("--task", default="move the arm around randomly", help="Fallback task string.")
    p.add_argument("--robot-type", default="hsrb", help="robot_type recorded in the dataset metadata.")
    p.add_argument("--max-age", type=float, default=0.5, help="Reject samples older than this [s].")
    p.add_argument("--min-frames", type=int, default=20, help="Skip episodes shorter than this.")
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all.")
    p.add_argument("--overwrite", action="store_true", help="Delete an existing output dataset.")
    p.add_argument("--topics-json", type=pathlib.Path, default=None, help="JSON overriding the topic names.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)

    topics = dict(DEFAULT_TOPICS)
    if args.topics_json:
        topics.update(json.loads(args.topics_json.read_text()))

    if not args.bag.exists():
        logger.error("bag not found: %s", args.bag)
        return 1

    series = read_bag(args.bag, topics)
    episodes = segment_episodes(series, args.task)
    logger.info("found %d episode(s)", len(episodes))
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    episodes_frames: List[Tuple[Episode, List[Dict[str, Any]]]] = []
    for episode in episodes:
        frames = build_frames(
            series, episode, fps=args.fps, image_order=args.image_order, max_age_s=args.max_age
        )
        if len(frames) < args.min_frames:
            logger.warning("episode %d has only %d frames; skipping", episode.index, len(frames))
            continue
        episodes_frames.append((episode, frames))

    if not episodes_frames:
        logger.error("no usable episodes")
        return 1

    out = write_lerobot(
        episodes_frames,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        overwrite=args.overwrite,
        robot_type=args.robot_type,
    )
    total = sum(len(f) for _, f in episodes_frames)
    logger.info("done: %d episodes / %d frames -> %s", len(episodes_frames), total, out)
    logger.info("next: compute norm stats, then train with dataset.repo_id=%s", args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
