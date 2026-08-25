#!/usr/bin/env python3
"""Drive the HSR with smooth random motions so that a data-collection bag can be
recorded in simulation.

The motion is an Ornstein-Uhlenbeck random walk per joint, clamped to a safe
sub-range of the URDF limits and rate limited, which keeps the trajectories
physically plausible (and therefore useful as training data) instead of the
white noise a naive random target would produce.

Commands go out on exactly the same topics the openpi policy uses, and
``/control_mode`` is published as ``auto`` while an episode is running, so a bag
recorded here has the same structure as one recorded on the real robot.

Episodes are managed automatically: ``episode_duration`` seconds of motion,
then the arm is driven back to the start pose for ``reset_duration`` seconds
before the next episode starts. ``~/start`` and ``~/stop`` (std_srvs/Trigger)
control it manually.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hsr_openpi.hsr_env import command_qos, sensor_qos

# URDF limits of hsrb4s, with a safety margin so the random walk never slams into
# a hard stop or folds the arm into the body.
#   joint: (low, high, ou_sigma [rad or m per sqrt(s)], max_rate [unit/s])
JOINT_RANGES: Dict[str, tuple] = {
    "arm_lift_joint": (0.0, 0.50, 0.25, 0.15),      # URDF 0.00 .. 0.69
    "arm_flex_joint": (-2.00, -0.10, 0.90, 0.90),   # URDF -2.62 .. 0.00
    "arm_roll_joint": (-1.80, 1.80, 0.90, 1.20),    # URDF -2.09 .. 3.84
    "wrist_flex_joint": (-1.80, 0.40, 0.90, 1.10),  # URDF -1.92 .. 1.22
    "wrist_roll_joint": (-1.50, 1.50, 0.90, 1.10),  # URDF -1.92 .. 3.67
    "head_pan_joint": (-1.00, 1.00, 0.50, 0.70),    # URDF -3.84 .. 1.75
    "head_tilt_joint": (-0.90, 0.10, 0.35, 0.60),   # URDF -1.57 .. 0.52
    "hand_motor_joint": (0.0, 1.20, 1.20, 1.60),    # URDF -0.80 .. 1.24
}

ARM_JOINTS = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
HAND_JOINTS = ["hand_motor_joint"]

START_POSE: Dict[str, float] = {
    "arm_lift_joint": 0.0,
    "arm_flex_joint": -0.3,
    "arm_roll_joint": 0.0,
    "wrist_flex_joint": -1.2,
    "wrist_roll_joint": 0.0,
    "head_pan_joint": 0.0,
    "head_tilt_joint": -0.4,
    "hand_motor_joint": 1.0,
}


class OrnsteinUhlenbeck:
    """Mean reverting random walk, clamped and rate limited."""

    def __init__(self, *, low: float, high: float, sigma: float, max_rate: float, theta: float, rng: np.random.Generator):
        self.low = float(low)
        self.high = float(high)
        self.mid = 0.5 * (self.low + self.high)
        self.sigma = float(sigma)
        self.max_rate = float(max_rate)
        self.theta = float(theta)
        self.rng = rng
        self.value = self.mid

    def reset(self, value: Optional[float] = None) -> None:
        self.value = self.mid if value is None else float(np.clip(value, self.low, self.high))

    def step(self, dt: float) -> float:
        drift = self.theta * (self.mid - self.value) * dt
        noise = self.sigma * np.sqrt(dt) * self.rng.standard_normal()
        delta = float(np.clip(drift + noise, -self.max_rate * dt, self.max_rate * dt))
        self.value = float(np.clip(self.value + delta, self.low, self.high))
        return self.value


class RandomMotion(Node):
    def __init__(self) -> None:
        super().__init__("hsr_random_motion")

        self.declare_parameter("update_freq", 10.0)
        self.declare_parameter("episode_duration", 30.0)
        self.declare_parameter("reset_duration", 6.0)
        self.declare_parameter("num_episodes", 0)  # 0 = run until stopped
        self.declare_parameter("seed", 0)
        self.declare_parameter("theta", 0.6)
        self.declare_parameter("move_base", True)
        self.declare_parameter("base_linear_scale", 0.12)   # [m/s]
        self.declare_parameter("base_angular_scale", 0.45)  # [rad/s]
        self.declare_parameter("auto_start", True)
        self.declare_parameter("task", "move the arm around randomly")
        self.declare_parameter("arm_command_topic", "/arm_trajectory_controller/joint_trajectory")
        self.declare_parameter("head_command_topic", "/head_trajectory_controller/joint_trajectory")
        self.declare_parameter("gripper_command_topic", "/gripper_controller/joint_trajectory")
        self.declare_parameter("base_command_topic", "/omni_base_controller/cmd_vel")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("control_mode_topic", "/control_mode")
        self.declare_parameter("episode_topic", "~/episode")

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.update_freq = float(g("update_freq"))
        self.dt = 1.0 / self.update_freq
        self.episode_duration = float(g("episode_duration"))
        self.reset_duration = float(g("reset_duration"))
        self.num_episodes = int(g("num_episodes"))
        self.move_base = bool(g("move_base"))
        self.base_linear_scale = float(g("base_linear_scale"))
        self.base_angular_scale = float(g("base_angular_scale"))
        self.task = str(g("task"))

        rng = np.random.default_rng(int(g("seed")))
        theta = float(g("theta"))
        self.walkers = {
            name: OrnsteinUhlenbeck(
                low=low, high=high, sigma=sigma, max_rate=max_rate, theta=theta, rng=rng
            )
            for name, (low, high, sigma, max_rate) in JOINT_RANGES.items()
        }
        self.base_walkers = [
            OrnsteinUhlenbeck(low=-1.0, high=1.0, sigma=1.5, max_rate=3.0, theta=1.2, rng=rng) for _ in range(3)
        ]
        self.rng = rng

        self.arm_pub = self.create_publisher(JointTrajectory, g("arm_command_topic"), command_qos())
        self.head_pub = self.create_publisher(JointTrajectory, g("head_command_topic"), command_qos())
        self.gripper_pub = self.create_publisher(JointTrajectory, g("gripper_command_topic"), command_qos())
        self.base_pub = self.create_publisher(Twist, g("base_command_topic"), command_qos())
        self.control_mode_pub = self.create_publisher(String, g("control_mode_topic"), command_qos(10))
        self.episode_pub = self.create_publisher(String, str(g("episode_topic")), command_qos(10))

        self._lock = threading.Lock()
        self.joint_state: Optional[Dict[str, float]] = None
        self.create_subscription(JointState, g("joint_states_topic"), self._joint_state_cb, sensor_qos(10))

        self.create_service(Trigger, "~/start", self._start_srv)
        self.create_service(Trigger, "~/stop", self._stop_srv)

        self.running = bool(g("auto_start"))
        self._finished = threading.Event()
        self.episode_index = 0
        self.state = "reset"  # "reset" | "run"
        self.phase_elapsed = 0.0
        self._announced_episode = -1

        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            "hsr_random_motion started: "
            f"update_freq={self.update_freq} episode_duration={self.episode_duration}s "
            f"reset_duration={self.reset_duration}s num_episodes={self.num_episodes or 'inf'} "
            f"move_base={self.move_base} task='{self.task}'"
        )

    # ------------------------------------------------------------------ #
    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def _joint_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self.joint_state = {name: msg.position[i] for i, name in enumerate(msg.name)}

    def _start_srv(self, request, response):
        self.running = True
        response.success = True
        response.message = "random motion started"
        return response

    def _stop_srv(self, request, response):
        self.running = False
        self._publish_control_mode("stopped")
        self.base_pub.publish(Twist())
        response.success = True
        response.message = "random motion stopped"
        return response

    # ------------------------------------------------------------------ #
    def _publish_control_mode(self, mode: str) -> None:
        self.control_mode_pub.publish(String(data=mode))

    def _trajectory(self, names: List[str], values: List[float], duration_s: float) -> JointTrajectory:
        traj = JointTrajectory()
        # Stamp the header so the bag converter can place the command on the
        # simulation timeline instead of the recorder's wall clock.
        # Leave header.stamp at zero: joint_trajectory_controller reads that as
        # "start now". Stamping with this node's clock makes the trajectory's
        # start an absolute sim time, and a Python node's /clock handling lags
        # behind the controller running inside Gazebo -- once that lag exceeds
        # the 0.2 s duration used here, every trajectory arrives already expired
        # and the controller drops the lot ("Received trajectory with non-zero
        # start time that ends in the past"). The arm then creeps instead of
        # moving and every pick fails, with the reason buried in the simulator's
        # log. hsr_env._trajectory has always left it at zero.
        traj.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in values]
        point.time_from_start = Duration(seconds=duration_s).to_msg()
        traj.points = [point]
        return traj

    def _send(self, targets: Dict[str, float], duration_s: float) -> None:
        self.arm_pub.publish(self._trajectory(ARM_JOINTS, [targets[j] for j in ARM_JOINTS], duration_s))
        self.head_pub.publish(self._trajectory(HEAD_JOINTS, [targets[j] for j in HEAD_JOINTS], duration_s))
        self.gripper_pub.publish(self._trajectory(HAND_JOINTS, [targets[j] for j in HAND_JOINTS], 1.0))

    def _begin_episode(self) -> None:
        with self._lock:
            current = dict(self.joint_state or {})
        for name, walker in self.walkers.items():
            walker.reset(current.get(name, START_POSE.get(name)))
        for walker in self.base_walkers:
            walker.reset(0.0)
        self.episode_pub.publish(String(data=f"start {self.episode_index} {self.task}"))
        self.get_logger().info(f"episode {self.episode_index} started ({self.episode_duration:.0f}s)")

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if not self.running:
            return

        if self.num_episodes and self.episode_index >= self.num_episodes:
            if self.running:
                self.get_logger().info(f"finished {self.episode_index} episodes; stopping.")
                self.running = False
                self._publish_control_mode("stopped")
                self.base_pub.publish(Twist())
                self.episode_pub.publish(String(data="done"))
                # Exit the process so that the launch file can shut the bag
                # recorder down cleanly. rosbag2 only writes metadata.yaml on
                # SIGINT; killing it leaves an unreadable bag.
                self._finished.set()
            return

        self.phase_elapsed += self.dt

        if self.state == "reset":
            # Hold the start pose, keep the base still, and do not mark the data
            # as an active episode.
            self._publish_control_mode("reset")
            self._send(START_POSE, max(self.reset_duration * 0.5, 1.0))
            self.base_pub.publish(Twist())
            if self.phase_elapsed >= self.reset_duration:
                self.state = "run"
                self.phase_elapsed = 0.0
                self._begin_episode()
            return

        # --- running -------------------------------------------------- #
        self._publish_control_mode("auto")
        targets = {name: walker.step(self.dt) for name, walker in self.walkers.items()}
        self._send(targets, self.dt * 1.5)

        twist = Twist()
        if self.move_base:
            twist.linear.x = self.base_linear_scale * self.base_walkers[0].step(self.dt)
            twist.linear.y = self.base_linear_scale * self.base_walkers[1].step(self.dt)
            twist.angular.z = self.base_angular_scale * self.base_walkers[2].step(self.dt)
        self.base_pub.publish(twist)

        if self.phase_elapsed >= self.episode_duration:
            self.episode_pub.publish(String(data=f"end {self.episode_index}"))
            self.get_logger().info(f"episode {self.episode_index} finished")
            self.episode_index += 1
            self.state = "reset"
            self.phase_elapsed = 0.0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RandomMotion()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.base_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
