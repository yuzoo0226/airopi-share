#!/usr/bin/env python3
"""Move the HSR to a known start pose before a rollout (ROS 2 port of reset_pose.py).

The pose is read from a YAML file (``pose_file`` parameter). When the file is
missing the built-in default is used, which puts the arm in the HSR "go" posture
with the head tilted down towards a table.
"""

from __future__ import annotations

import os
from typing import Dict

import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hsr_openpi.hsr_env import command_qos

ARM_JOINTS = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
HAND_JOINTS = ["hand_motor_joint"]

DEFAULT_POSE: Dict[str, float] = {
    "arm_lift_joint": 0.0,
    "arm_flex_joint": 0.0,
    "arm_roll_joint": -1.57,
    "wrist_flex_joint": -1.57,
    "wrist_roll_joint": 0.0,
    "head_pan_joint": 0.0,
    "head_tilt_joint": -0.4,
    "hand_motor_joint": 1.2,
}


class ResetPose(Node):
    def __init__(self) -> None:
        super().__init__("reset_pose")
        self.declare_parameter("pose_file", "")
        self.declare_parameter("duration", 5.0)
        self.declare_parameter("arm_command_topic", "/arm_trajectory_controller/joint_trajectory")
        self.declare_parameter("head_command_topic", "/head_trajectory_controller/joint_trajectory")
        self.declare_parameter("gripper_command_topic", "/gripper_controller/joint_trajectory")

        self.duration = float(self.get_parameter("duration").value)
        self.arm_pub = self.create_publisher(
            JointTrajectory, self.get_parameter("arm_command_topic").value, command_qos()
        )
        self.head_pub = self.create_publisher(
            JointTrajectory, self.get_parameter("head_command_topic").value, command_qos()
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory, self.get_parameter("gripper_command_topic").value, command_qos()
        )

    def load_pose(self) -> Dict[str, float]:
        pose_file = str(self.get_parameter("pose_file").value or "")
        if pose_file and os.path.exists(pose_file):
            with open(pose_file, "r") as f:
                data = yaml.safe_load(f) or {}
            pose = dict(DEFAULT_POSE)
            pose.update({k: float(v) for k, v in data.items() if k in DEFAULT_POSE})
            self.get_logger().info(f"Loaded pose from {pose_file}")
            return pose
        if pose_file:
            self.get_logger().warn(f"Pose file '{pose_file}' not found; using the built-in default pose.")
        return dict(DEFAULT_POSE)

    def publish_trajectory(self, pub, joint_names, pose: Dict[str, float]) -> None:
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(pose[j]) for j in joint_names]
        point.velocities = [0.0] * len(joint_names)
        point.time_from_start = Duration(seconds=self.duration).to_msg()
        traj.points = [point]
        pub.publish(traj)

    def send(self) -> None:
        pose = self.load_pose()
        self.get_logger().info(f"Resetting to pose: {pose}")
        self.publish_trajectory(self.arm_pub, ARM_JOINTS, pose)
        self.publish_trajectory(self.head_pub, HEAD_JOINTS, pose)
        self.publish_trajectory(self.gripper_pub, HAND_JOINTS, pose)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ResetPose()
    # Give the publishers a moment to match with the controllers.
    end = node.get_clock().now() + Duration(seconds=1.5)
    while rclpy.ok() and node.get_clock().now() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.send()
    end = node.get_clock().now() + Duration(seconds=node.duration + 1.0)
    while rclpy.ok() and node.get_clock().now() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info("Done.")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
