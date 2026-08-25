#!/usr/bin/env python3
"""Does the head actually end up looking at the gripper?

The gaze servo in `pick_task` turns the head by the angle between the camera's
optical axis and `hand_palm_link`. That is only correct if `head_pan_joint` turns
the camera the way this assumes and `head_rgbd_sensor_rgb_frame` really follows
the optical convention (+x right, +y down, +z forward) -- get either backwards
and the head walks to its limit instead of converging, which is easy to miss in
recorded data and impossible to fix afterwards.

So drive it against the running simulator and watch the error:

    ros2 run ... # not installed; run directly inside the sim container
    python3 deploy/hsr_openpi_ros2/tools/check_head_gaze.py --seconds 20

The arm is first moved to the pick configuration so the gripper is out in front
of the robot where the head has to turn to find it. PASS means the residual
angle fell below --tolerance and stayed there.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
PICK_ARM = [0.20, -math.pi / 2, 0.0, -math.pi / 2, 0.0]


def command_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class GazeCheck(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("check_head_gaze")
        self.args = args
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])

        self.arm_pub = self.create_publisher(
            JointTrajectory, "/arm_trajectory_controller/joint_trajectory", command_qos()
        )
        self.head_pub = self.create_publisher(
            JointTrajectory, "/head_trajectory_controller/joint_trajectory", command_qos()
        )
        self.joint_state: dict = {}
        self.create_subscription(JointState, "/joint_states", self._joints, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def _joints(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self.joint_state[name] = position

    def _traj(self, names, positions, duration_s: float) -> JointTrajectory:
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        point.time_from_start = Duration(seconds=float(duration_s)).to_msg()
        traj.points.append(point)
        return traj

    def gaze_error(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.camera_frame, self.args.gaze_frame, rclpy.time.Time()
            )
        except TransformException:
            return None
        t = tf.transform.translation
        if t.z <= 1e-3:
            return None
        return math.atan2(t.x, t.z), math.atan2(t.y, t.z)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--max-rate", type=float, default=1.2)
    ap.add_argument("--deadband", type=float, default=0.010)
    ap.add_argument("--tolerance", type=float, default=0.05, help="radians")
    ap.add_argument("--camera-frame", default="head_rgbd_sensor_rgb_frame")
    ap.add_argument("--gaze-frame", default="hand_palm_link")
    ap.add_argument("--initial-tilt", type=float, default=-0.45)
    args = ap.parse_args(argv)

    rclpy.init()
    node = GazeCheck(args)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    import threading

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    dt = 1.0 / args.rate
    rate = node.create_rate(args.rate)

    # settle, then reach out so the gripper is somewhere the head must turn to see
    for _ in range(10):
        rate.sleep()
    node.arm_pub.publish(node._traj(ARM_JOINTS, PICK_ARM, 2.0))
    head_cmd = [0.0, args.initial_tilt]
    node.head_pub.publish(node._traj(HEAD_JOINTS, head_cmd, 1.0))
    for _ in range(int(3.0 * args.rate)):
        rate.sleep()

    pan_limits, tilt_limits = (-3.83, 1.75), (-1.57, 0.52)
    history = []
    steps = int(args.seconds * args.rate)
    print(f"{'t':>6s} {'yaw_err':>9s} {'pitch_err':>9s} {'pan_cmd':>9s} {'tilt_cmd':>9s} {'pan':>8s} {'tilt':>8s}")
    for i in range(steps):
        error = node.gaze_error()
        if error is not None:
            limit = args.max_rate * dt
            for j, (err, lo_hi) in enumerate(zip(error, (pan_limits, tilt_limits))):
                if abs(err) < args.deadband:
                    continue
                step = float(np.clip(-args.gain * err, -limit, limit))
                head_cmd[j] = float(np.clip(head_cmd[j] + step, lo_hi[0], lo_hi[1]))
            history.append(error)
        node.head_pub.publish(node._traj(HEAD_JOINTS, head_cmd, dt * 2))
        if i % int(args.rate / 2) == 0 and error is not None:
            pan = node.joint_state.get("head_pan_joint", float("nan"))
            tilt = node.joint_state.get("head_tilt_joint", float("nan"))
            print(
                f"{i * dt:6.1f} {error[0]:9.4f} {error[1]:9.4f} "
                f"{head_cmd[0]:9.4f} {head_cmd[1]:9.4f} {pan:8.4f} {tilt:8.4f}"
            )
        rate.sleep()

    rclpy.shutdown()
    thread.join(timeout=2.0)

    if not history:
        print("\nFAIL: the transform never resolved -- wrong frame names?")
        return 1
    final = np.abs(np.array(history[-int(2.0 * args.rate):], dtype=float))
    worst = float(final.max())
    print(f"\nresidual over the last 2 s: yaw {final[:, 0].mean():.4f}  pitch {final[:, 1].mean():.4f}  worst {worst:.4f}")
    if worst <= args.tolerance:
        print(f"PASS: the head converged on {args.gaze_frame} (tolerance {args.tolerance})")
        return 0
    start = float(np.abs(np.array(history[0], dtype=float)).max())
    print(f"FAIL: started at {start:.4f} rad and ended at {worst:.4f} rad")
    print("      a growing error means a sign is inverted; a stuck one means the head hit a limit")
    return 1


if __name__ == "__main__":
    sys.exit(main())
