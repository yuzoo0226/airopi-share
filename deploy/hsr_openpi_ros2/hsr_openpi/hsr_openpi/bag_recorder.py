#!/usr/bin/env python3
"""Episode-aware ROS 2 bag recorder, meant to run on the HSR itself.

Teleoperation with `hsr_leader_teleop
<https://github.com/Hibikino-Musashi-Home/hsr_leader_teleop>`_ publishes on
exactly the topics this deployment uses (``/arm_trajectory_controller/
joint_trajectory``, ``/omni_base_controller/cmd_vel``, ``/gripper_controller/
grasp`` and friends), so a bag recorded while teleoperating converts into
training data with ``tools/rosbag2_to_lerobot.py`` unchanged.

What this node adds over plain ``ros2 bag record``:

* **episode markers** - it publishes ``start <index> <task>`` / ``end <index>``
  on ``~/episode``, the same contract the simulator's pick task uses, so the
  converter can segment the bag into episodes;
* **start/stop control** while the recorder keeps running, via
  ``~/start_episode`` (StringTrigger, the message is the task string) and
  ``~/stop_episode`` (Trigger); ``~/discard_episode`` drops the current one;
* **a clean shutdown** - ``ros2 bag record`` is stopped with SIGINT, which is
  the only way it writes ``metadata.yaml``. A killed recorder leaves a bag that
  neither ``ros2 bag info`` nor the converter can open.

Bags are written under ``output_dir`` on whatever machine the node runs on, so
launching it on the robot keeps the data on the robot.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import shutil
import signal
import subprocess
import threading
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from hsr_openpi.hsr_env import command_qos

try:
    from hsr_openpi_msgs.srv import StringTrigger

    _HAS_STRING_TRIGGER = True
except ImportError:  # pragma: no cover
    StringTrigger = None  # type: ignore[assignment]
    _HAS_STRING_TRIGGER = False


# The real robot publishes under /hsrb; the simulator does not. Both sets are
# offered so the same node can be tested against Gazebo before going near
# hardware.
REAL_ROBOT_TOPICS = [
    "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed",
    "/hsrb/head_rgbd_sensor/rgb/camera_info",
    "/hsrb/hand_camera/image_raw/compressed",
    "/hsrb/hand_camera/camera_info",
    "/hsrb/joint_states",
    "/hsrb/arm_trajectory_controller/joint_trajectory",
    "/hsrb/head_trajectory_controller/joint_trajectory",
    "/hsrb/gripper_controller/joint_trajectory",
    "/hsrb/omni_base_controller/cmd_vel",
    "/hsrb/omni_base_controller/wheel_odom",
    "/hsrb/wrist_wrench/raw",
    "/tf",
    "/tf_static",
    "/control_mode",
]

SIM_TOPICS = [
    "/clock",
    "/head_rgbd_sensor/rgb/image_rect_color/compressed",
    "/head_rgbd_sensor/rgb/camera_info",
    "/hand_camera/image_raw/compressed",
    "/hand_camera/camera_info",
    "/joint_states",
    "/arm_trajectory_controller/joint_trajectory",
    "/head_trajectory_controller/joint_trajectory",
    "/gripper_controller/joint_trajectory",
    "/omni_base_controller/cmd_vel",
    "/odom",
    "/tf",
    "/tf_static",
    "/control_mode",
]


class BagRecorder(Node):
    def __init__(self) -> None:
        super().__init__("hsr_bag_recorder")

        p = self.declare_parameter
        p("output_dir", str(pathlib.Path.home() / "hsr_bags"))
        p("bag_name", "")  # empty -> timestamped
        p("profile", "real")  # "real" | "sim" | "custom"
        p("topics", [])  # used when profile == "custom"
        p("storage", "mcap")
        p("max_bagfile_size", 0)  # bytes, 0 = single file
        p("compression", "")  # "" | "zstd"
        p("task", "teleoperation")
        p("auto_start", False)  # start the first episode immediately
        p("episode_topic", "~/episode")
        p("min_episode_seconds", 1.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        profile = str(g("profile")).lower()
        if profile == "real":
            self.topics: List[str] = list(REAL_ROBOT_TOPICS)
        elif profile == "sim":
            self.topics = list(SIM_TOPICS)
        else:
            self.topics = [str(t) for t in (g("topics") or [])]
        if not self.topics:
            raise RuntimeError("no topics to record (set profile or topics)")

        self.output_dir = pathlib.Path(str(g("output_dir"))).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = str(g("bag_name")) or datetime.datetime.now().strftime("teleop_%Y%m%d_%H%M%S")
        self.bag_path = self.output_dir / name
        self.task = str(g("task"))
        self.min_episode_seconds = float(g("min_episode_seconds"))

        self.episode_pub = self.create_publisher(String, str(g("episode_topic")), command_qos(10))

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._episode_index = 0
        self._episode_open = False
        self._episode_start_ns = 0
        self._episodes: List[dict] = []

        if _HAS_STRING_TRIGGER:
            self.create_service(StringTrigger, "~/start_episode", self._start_srv)
        else:
            self.get_logger().warn(
                "hsr_openpi_msgs is unavailable: ~/start_episode falls back to std_srvs/Trigger"
            )
            self.create_service(Trigger, "~/start_episode", self._start_plain_srv)
        self.create_service(Trigger, "~/stop_episode", self._stop_srv)
        self.create_service(Trigger, "~/discard_episode", self._discard_srv)
        self.create_subscription(String, "~/task", self._task_cb, command_qos(10))

        self._start_recorder()
        if bool(g("auto_start")):
            self._open_episode(self.task)

        self.get_logger().info(
            f"recording {len(self.topics)} topics (profile={profile}) into {self.bag_path}\n"
            f"  start: ros2 service call {self.get_name()}/start_episode "
            f"hsr_openpi_msgs/srv/StringTrigger \"{{message: 'pick up the bottle'}}\"\n"
            f"  stop : ros2 service call {self.get_name()}/stop_episode std_srvs/srv/Trigger"
        )

    # ------------------------------------------------------------------ #
    def _start_recorder(self) -> None:
        if shutil.which("ros2") is None:
            raise RuntimeError("the ros2 CLI is not on PATH")
        cmd = ["ros2", "bag", "record", "-s", str(self.get_parameter("storage").value),
               "-o", str(self.bag_path)]
        size = int(self.get_parameter("max_bagfile_size").value)
        if size > 0:
            cmd += ["-b", str(size)]
        compression = str(self.get_parameter("compression").value)
        if compression:
            cmd += ["--compression-mode", "file", "--compression-format", compression]
        cmd += self.topics
        self.get_logger().info("starting: " + " ".join(cmd))
        # New process group so SIGINT can be sent to the recorder alone.
        self._proc = subprocess.Popen(cmd, start_new_session=True)

    def stop_recorder(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        if self._episode_open:
            self._close_episode(keep=True)
        self.get_logger().info("stopping the recorder (SIGINT, so metadata.yaml gets written)")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=30)
        except (ProcessLookupError, PermissionError):
            pass
        except subprocess.TimeoutExpired:
            self.get_logger().warn("recorder did not exit within 30 s; terminating")
            proc.terminate()
        self._write_index()

    # ------------------------------------------------------------------ #
    def _open_episode(self, task: str) -> bool:
        if self._episode_open:
            self.get_logger().warn("an episode is already open; ignoring")
            return False
        self.task = task or self.task
        self._episode_start_ns = self.get_clock().now().nanoseconds
        self.episode_pub.publish(String(data=f"start {self._episode_index} {self.task}"))
        self._episode_open = True
        self.get_logger().info(f"episode {self._episode_index} started: {self.task!r}")
        return True

    def _close_episode(self, *, keep: bool) -> bool:
        if not self._episode_open:
            self.get_logger().warn("no episode is open; ignoring")
            return False
        duration = (self.get_clock().now().nanoseconds - self._episode_start_ns) * 1e-9
        self._episode_open = False
        if not keep:
            self.episode_pub.publish(String(data=f"discard {self._episode_index}"))
            self.get_logger().info(f"episode {self._episode_index} discarded after {duration:.1f}s")
            return True
        if duration < self.min_episode_seconds:
            self.episode_pub.publish(String(data=f"discard {self._episode_index}"))
            self.get_logger().warn(
                f"episode {self._episode_index} only lasted {duration:.1f}s "
                f"(< min_episode_seconds); discarded"
            )
            return True
        self.episode_pub.publish(String(data=f"end {self._episode_index}"))
        self._episodes.append(
            {"episode": self._episode_index, "task": self.task, "duration_s": round(duration, 2)}
        )
        self.get_logger().info(
            f"episode {self._episode_index} ended after {duration:.1f}s "
            f"({len(self._episodes)} kept)"
        )
        self._episode_index += 1
        self._write_index()
        return True

    def _write_index(self) -> None:
        try:
            path = self.bag_path.with_suffix(".episodes.json")
            path.write_text(json.dumps({"bag": str(self.bag_path), "episodes": self._episodes}, indent=2))
        except OSError as e:  # pragma: no cover
            self.get_logger().warn(f"could not write the episode index: {e}")

    # -- service / topic callbacks --------------------------------------- #
    def _start_srv(self, request, response):
        ok = self._open_episode(request.message)
        response.success = ok
        response.message = f"episode {self._episode_index} started" if ok else "an episode is already open"
        return response

    def _start_plain_srv(self, request, response):
        ok = self._open_episode(self.task)
        response.success = ok
        response.message = f"episode {self._episode_index} started" if ok else "an episode is already open"
        return response

    def _stop_srv(self, request, response):
        ok = self._close_episode(keep=True)
        response.success = ok
        response.message = "episode closed" if ok else "no episode is open"
        return response

    def _discard_srv(self, request, response):
        ok = self._close_episode(keep=False)
        response.success = ok
        response.message = "episode discarded" if ok else "no episode is open"
        return response

    def _task_cb(self, msg: String) -> None:
        self.task = msg.data
        self.get_logger().info(f"task set to {self.task!r}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BagRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recorder()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
