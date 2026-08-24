"""ROS 2 interface to the HSR (real robot or Ignition Gazebo simulation).

This is the ROS 2 port of the ``HSREnv`` class from the ROS 1 deployment node.
Every topic / action name is a ROS parameter because the simulator and the real
robot do not agree on them:

===================  ==========================================  ===========================================
purpose              real robot (ROS 1 / hsrb bringup)           Ignition Gazebo (hsrb_gazebo_bringup)
===================  ==========================================  ===========================================
head camera          /hsrb/head_rgbd_sensor/rgb/                 /head_rgbd_sensor/rgb/image_rect_color
                     image_rect_color/compressed
hand camera          /hsrb/hand_camera/image_raw/compressed      /hand_camera/image_raw
joint states         /hsrb/joint_states                          /joint_states
arm command          /hsrb/arm_trajectory_controller/command     /arm_trajectory_controller/joint_trajectory
head command         /hsrb/head_trajectory_controller/command    /head_trajectory_controller/joint_trajectory
gripper command      /hsrb/gripper_controller/command            /gripper_controller/joint_trajectory
base command         /hsrb/command_velocity (Twist)              /omni_base_controller/cmd_vel (Twist)
gripper grasp        /hsrb/gripper_controller/grasp              /gripper_controller/grasp
===================  ==========================================  ===========================================
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hsr_openpi.action_utils import MODE_CONTINUOUS, MODE_DISCRETE, MODE_HYBRID, GRIPPER_MODES

try:  # tmc_control_msgs is only available when the HSR stack is sourced.
    from tmc_control_msgs.action import GripperApplyEffort

    _HAS_GRIPPER_ACTION = True
except ImportError:  # pragma: no cover - depends on the runtime workspace
    GripperApplyEffort = None  # type: ignore[assignment]
    _HAS_GRIPPER_ACTION = False


# Image channel order (see docs/ros2_deploy.md 6.1).
#
# The source order depends on where the frame comes from:
#   * CompressedImage  -> cv2.imdecode returns BGR
#   * Image "rgb8"     -> RGB   (what Ignition Gazebo publishes)
#   * Image "bgr8"     -> BGR
# so the node normalises everything to a single order before handing the frame
# to the policy, selected by the `policy_image_order` parameter. Without this,
# the simulator would feed RGB while the real robot feeds BGR.
#
# The default is "bgr": the ROS 1 deployment node and the ROS 2 client of the
# ICRA evaluation runtime both pass the raw cv2.imdecode output, so that is the
# order the released checkpoints were deployed with — even though the dataset
# conversion pipeline ends up storing RGB.
IMAGE_ORDER_RGB = "rgb"
IMAGE_ORDER_BGR = "bgr"
IMAGE_ORDERS = [IMAGE_ORDER_BGR, IMAGE_ORDER_RGB]

# encoding -> (channel count, channel order of the decoded buffer)
_ENCODING_INFO = {
    "rgb8": (3, IMAGE_ORDER_RGB),
    "bgr8": (3, IMAGE_ORDER_BGR),
    "rgba8": (4, IMAGE_ORDER_RGB),
    "bgra8": (4, IMAGE_ORDER_BGR),
    "mono8": (1, None),
    "8UC1": (1, None),
    "8UC3": (3, IMAGE_ORDER_BGR),
}


def sensor_qos(depth: int = 1) -> QoSProfile:
    """Best-effort QoS: compatible with both reliable and best-effort publishers."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def command_qos(depth: int = 1) -> QoSProfile:
    """Reliable QoS for command topics (accepted by best-effort subscribers too)."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class HSREnv:
    """Collects HSR observations and forwards openpi actions to the controllers."""

    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    GRIPPER_CLOSE_THRESHOLD = 0.25
    GRIPPER_OPEN_POSITION = 1.239183768915874

    JOINT_STATE_NAMES: List[str] = [
        "arm_lift_joint",
        "arm_flex_joint",
        "arm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        "hand_motor_joint",
        "head_pan_joint",
        "head_tilt_joint",
    ]
    ARM_ACTION_NAMES: List[str] = [
        "arm_lift_joint",
        "arm_flex_joint",
        "arm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]
    HEAD_ACTION_NAMES: List[str] = ["head_pan_joint", "head_tilt_joint"]
    BASE_ACTION_NAMES: List[str] = ["base_x", "base_y", "base_theta"]

    def __init__(self, node: Node, *, update_freq: float = 10.0):
        self._node = node
        self._logger = node.get_logger()
        self.update_freq = float(update_freq)

        p = node.get_parameter
        self.head_image_topic: str = p("head_image_topic").value
        self.hand_image_topic: str = p("hand_image_topic").value
        self.joint_states_topic: str = p("joint_states_topic").value
        self.control_mode_topic: str = p("control_mode_topic").value
        self.arm_command_topic: str = p("arm_command_topic").value
        self.head_command_topic: str = p("head_command_topic").value
        self.gripper_command_topic: str = p("gripper_command_topic").value
        self.base_command_topic: str = p("base_command_topic").value
        self.gripper_grasp_action: str = p("gripper_grasp_action").value
        self.gripper_mode: str = p("gripper_mode").value
        self.gripper_effort: float = float(p("gripper_effort").value)
        self.policy_image_order: str = str(p("policy_image_order").value).lower()
        if self.policy_image_order not in IMAGE_ORDERS:
            self._logger.warn(
                f"Unknown policy_image_order '{self.policy_image_order}'. Falling back to "
                f"'{IMAGE_ORDER_BGR}'. Available: {', '.join(IMAGE_ORDERS)}"
            )
            self.policy_image_order = IMAGE_ORDER_BGR
        self.require_control_mode: bool = bool(p("require_control_mode").value)
        self.control_mode_active_value: str = p("control_mode_active_value").value
        self.instruction: str = p("instruction").value

        if self.gripper_mode not in GRIPPER_MODES:
            self._logger.warn(
                f"Unknown gripper_mode '{self.gripper_mode}'. Falling back to '{MODE_CONTINUOUS}'. "
                f"Available: {', '.join(GRIPPER_MODES)}"
            )
            self.gripper_mode = MODE_CONTINUOUS

        # -- observation state (guarded by a lock: callbacks run in the executor
        #    thread while the control loop runs in the main thread) ---------- #
        self._lock = threading.Lock()
        self.head_rgb: Optional[np.ndarray] = None
        self.hand_rgb: Optional[np.ndarray] = None
        self.joint_state: Optional[np.ndarray] = None
        self.gripper_state: int = self.GRIPPER_OPEN
        self.control_mode: Optional[str] = None
        self._missing_joint_warned = False

        # -- publishers ------------------------------------------------------ #
        self.arm_pub = node.create_publisher(JointTrajectory, self.arm_command_topic, command_qos())
        self.head_pub = node.create_publisher(JointTrajectory, self.head_command_topic, command_qos())
        self.gripper_pub = node.create_publisher(JointTrajectory, self.gripper_command_topic, command_qos())
        self.base_pub = node.create_publisher(Twist, self.base_command_topic, command_qos())

        # -- gripper grasp action ------------------------------------------- #
        self.gripper_grasp_client = None
        if _HAS_GRIPPER_ACTION and self.gripper_mode in (MODE_DISCRETE, MODE_HYBRID):
            self.gripper_grasp_client = ActionClient(node, GripperApplyEffort, self.gripper_grasp_action)
        elif self.gripper_mode in (MODE_DISCRETE, MODE_HYBRID):
            self._logger.warn(
                "tmc_control_msgs is not available: gripper_mode "
                f"'{self.gripper_mode}' falls back to '{MODE_CONTINUOUS}'."
            )
            self.gripper_mode = MODE_CONTINUOUS

        # -- subscriptions --------------------------------------------------- #
        self._make_image_subscription(self.head_image_topic, self._head_image_callback)
        self._make_image_subscription(self.hand_image_topic, self._hand_image_callback)
        node.create_subscription(JointState, self.joint_states_topic, self._joint_state_callback, sensor_qos(10))
        node.create_subscription(String, self.control_mode_topic, self._control_mode_callback, command_qos(10))

        self._logger.info(
            "HSREnv topics:\n"
            f"  head image : {self.head_image_topic}\n"
            f"  hand image : {self.hand_image_topic}\n"
            f"  joints     : {self.joint_states_topic}\n"
            f"  arm cmd    : {self.arm_command_topic}\n"
            f"  head cmd   : {self.head_command_topic}\n"
            f"  gripper cmd: {self.gripper_command_topic}\n"
            f"  base cmd   : {self.base_command_topic}\n"
            f"  grasp act  : {self.gripper_grasp_action if self.gripper_grasp_client else '<disabled>'}\n"
            f"  image order: {self.policy_image_order} (order handed to the policy)"
        )

    # ------------------------------------------------------------------ #
    # subscriptions
    # ------------------------------------------------------------------ #
    def _make_image_subscription(self, topic: str, callback):
        transport = self._node.get_parameter("image_transport").value
        use_compressed = transport == "compressed" or (transport == "auto" and topic.endswith("/compressed"))
        msg_type = CompressedImage if use_compressed else Image
        self._node.create_subscription(msg_type, topic, callback, sensor_qos())

    def _to_policy_order(self, image: np.ndarray, source_order: Optional[str]) -> np.ndarray:
        """Convert a decoded frame to the channel order the policy expects."""
        if source_order is None or source_order == self.policy_image_order:
            return image
        return image[:, :, ::-1]

    def _decode_compressed(self, msg: CompressedImage) -> np.ndarray:
        import cv2  # noqa: PLC0415

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # always BGR
        if image is None:
            raise ValueError("cv2.imdecode returned None")
        return self._to_policy_order(image, IMAGE_ORDER_BGR)

    def _decode_raw(self, msg: Image) -> np.ndarray:
        info = _ENCODING_INFO.get(msg.encoding)
        if info is None:
            raise ValueError(f"Unsupported image encoding '{msg.encoding}'")
        channels, source_order = info
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        image = buf.reshape(msg.height, msg.width, channels)
        if channels == 1:
            return np.repeat(image, 3, axis=2)
        image = image[:, :, :3]
        return self._to_policy_order(image, source_order)

    def _decode_image(self, msg) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            return np.ascontiguousarray(self._decode_compressed(msg))
        return np.ascontiguousarray(self._decode_raw(msg))

    def _head_image_callback(self, msg) -> None:
        try:
            image = self._decode_image(msg)
        except Exception as e:  # pragma: no cover - defensive
            self._logger.warn(f"Failed to decode head image: {e}")
            return
        with self._lock:
            self.head_rgb = image

    def _hand_image_callback(self, msg) -> None:
        try:
            image = self._decode_image(msg)
        except Exception as e:  # pragma: no cover - defensive
            self._logger.warn(f"Failed to decode hand image: {e}")
            return
        with self._lock:
            self.hand_rgb = image

    def _joint_state_callback(self, msg: JointState) -> None:
        try:
            index = {name: i for i, name in enumerate(msg.name)}
            joints = [msg.position[index[name]] for name in self.JOINT_STATE_NAMES]
        except KeyError as e:
            if not self._missing_joint_warned:
                self._logger.warn(
                    f"Joint {e} missing from {self.joint_states_topic}; available: {list(msg.name)}"
                )
                self._missing_joint_warned = True
            return
        with self._lock:
            self.joint_state = np.asarray(joints, dtype=np.float32)

    def _control_mode_callback(self, msg: String) -> None:
        with self._lock:
            self.control_mode = msg.data

    # ------------------------------------------------------------------ #
    # observation access
    # ------------------------------------------------------------------ #
    def set_instruction(self, instruction: str) -> None:
        with self._lock:
            self.instruction = instruction
        self._logger.info(f"Instruction updated: {instruction}")

    def set_control_mode(self, mode: str) -> None:
        with self._lock:
            self.control_mode = mode

    def reset_observation(self, *, reset_joint_state: bool = True) -> None:
        with self._lock:
            self.head_rgb = None
            self.hand_rgb = None
            if reset_joint_state:
                self.joint_state = None

    def get_observations(self) -> Optional[Dict[str, Any]]:
        """Full observation dict, or ``None`` while something is still missing."""
        with self._lock:
            if self.head_rgb is None or self.hand_rgb is None or self.joint_state is None:
                return None
            return {
                "head_rgb": self.head_rgb,
                "hand_rgb": self.hand_rgb,
                "joint_state": self.joint_state,
                "instruction": self.instruction,
                "gripper_state": self.gripper_state,
                "control_mode": self.control_mode,
            }

    def get_partial_observations(self) -> Optional[Dict[str, Any]]:
        """Joint state + instruction only (used while replaying a queued chunk)."""
        with self._lock:
            if self.joint_state is None:
                return None
            return {"joint_state": self.joint_state, "instruction": self.instruction}

    def missing_observations(self) -> List[str]:
        with self._lock:
            missing = []
            if self.head_rgb is None:
                missing.append(f"head_rgb({self.head_image_topic})")
            if self.hand_rgb is None:
                missing.append(f"hand_rgb({self.hand_image_topic})")
            if self.joint_state is None:
                missing.append(f"joint_state({self.joint_states_topic})")
            return missing

    def is_executable(self) -> bool:
        if not self.require_control_mode:
            return True
        with self._lock:
            return self.control_mode == self.control_mode_active_value

    # ------------------------------------------------------------------ #
    # command output
    # ------------------------------------------------------------------ #
    def _trajectory(self, joint_names: List[str], positions, duration_s: float) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        point.time_from_start = Duration(seconds=duration_s).to_msg()
        traj.points = [point]
        return traj

    def _send_grasp_goal(self, effort: float) -> None:
        if self.gripper_grasp_client is None:
            return
        if not self.gripper_grasp_client.server_is_ready():
            # wait_for_server would block the control loop, so only probe.
            self.gripper_grasp_client.wait_for_server(timeout_sec=0.0)
            if not self.gripper_grasp_client.server_is_ready():
                self._logger.warn(
                    f"Gripper grasp action server '{self.gripper_grasp_action}' is not available."
                )
                return
        goal = GripperApplyEffort.Goal()
        goal.effort = float(effort)
        # Fire and forget: the control loop must not block on the result.
        self.gripper_grasp_client.send_goal_async(goal)

    def _publish_gripper(self, gripper_value: float) -> None:
        value = float(np.clip(gripper_value, -0.1, 1.23))
        self.gripper_pub.publish(self._trajectory(["hand_motor_joint"], [value], 1.0))

    def execute_actions(self, action: np.ndarray) -> bool:
        """Send one 11-dim action to the robot.

        ``action`` layout::

            [arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll,
             gripper, head_pan, head_tilt, base_x, base_y, base_theta]

        Returns ``True`` when the command was actually published.
        """
        if not self.is_executable():
            return False

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 11:
            self._logger.warn(f"Expected an 11-dim action, got shape {action.shape}.")
            return False

        dt = 1.0 / self.update_freq / 2.0
        arm_traj = self._trajectory(self.ARM_ACTION_NAMES, action[:5], dt)
        head_traj = self._trajectory(self.HEAD_ACTION_NAMES, action[6:8], dt)

        twist = Twist()
        twist.linear.x = float(action[8])
        twist.linear.y = float(action[9])
        twist.angular.z = float(action[10])

        # -- gripper ------------------------------------------------------- #
        gripper_value = float(action[5])
        if self.gripper_mode == MODE_CONTINUOUS:
            self._publish_gripper(gripper_value)
        elif self.gripper_mode == MODE_DISCRETE:
            target = self.GRIPPER_CLOSE if gripper_value < self.GRIPPER_CLOSE_THRESHOLD else self.GRIPPER_OPEN
            if self.gripper_state != target:
                if target == self.GRIPPER_CLOSE:
                    self._send_grasp_goal(self.gripper_effort)
                else:
                    self._publish_gripper(self.GRIPPER_OPEN_POSITION)
                self.gripper_state = target
        elif self.gripper_mode == MODE_HYBRID:
            # Close through the force controlled action once the commanded
            # opening drops below the threshold, otherwise track it directly.
            if gripper_value < self.GRIPPER_CLOSE_THRESHOLD:
                if self.gripper_state != self.GRIPPER_CLOSE:
                    self._send_grasp_goal(self.gripper_effort)
                    self.gripper_state = self.GRIPPER_CLOSE
            else:
                self._publish_gripper(gripper_value)
                self.gripper_state = self.GRIPPER_OPEN

        self.arm_pub.publish(arm_traj)
        self.head_pub.publish(head_traj)
        self.base_pub.publish(twist)
        return True

    def stop_base(self) -> None:
        self.base_pub.publish(Twist())
