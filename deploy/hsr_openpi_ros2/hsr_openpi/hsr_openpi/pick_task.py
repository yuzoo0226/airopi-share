#!/usr/bin/env python3
"""Scripted pick-and-lift episodes in the Ignition Gazebo HSR simulator.

Every episode spawns one of ten objects at a random pose on a table, teleports
the robot to a randomised start pose that keeps the object in the head camera's
field of view, then drives the base, lowers the arm, closes the gripper and
lifts. Commands go out on the same topics the openpi policy uses, so a bag
recorded here converts into training data with
``deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py`` unchanged.

Grasping uses a *fixed* top-down arm configuration
(``arm_flex = wrist_flex = -pi/2``): the palm then points straight down at
(0.474, 0.078, 0.194 + arm_lift) in base_footprint, so aligning the gripper with
an object is pure base motion plus one prismatic joint, with no IK error. The
task the policy has to learn is therefore "look at the object, drive to it,
lower the arm the right amount, close, lift".

Success is judged two ways:

``ground_truth``
    the object's z from Gazebo's ``dynamic_pose`` stream rises above its resting
    height and stays there — used for the evaluation metric.
``gripper``
    the gripper stalls before reaching its commanded closing angle and the
    finger springs deflect — the same signal a real HSR gives, so the detector
    also works outside simulation.
"""

from __future__ import annotations

import json
import math
import pathlib
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hsr_openpi.gz_world import (
    FINGER_TIP_BELOW_PALM_OPEN,
    GRASP_FIX_NOTE,
    GzWorld,
    ObjectSpec,
    OBJECT_LIBRARY,
    PALM_OFFSET_X,
    PALM_OFFSET_Y,
    arm_lift_for_grasp,
    base_pose_for_grasp,
    table_sdf,
)
from hsr_openpi.hsr_env import command_qos, sensor_qos

try:
    from tmc_control_msgs.action import GripperApplyEffort

    _HAS_GRIPPER_ACTION = True
except ImportError:  # pragma: no cover
    GripperApplyEffort = None  # type: ignore[assignment]
    _HAS_GRIPPER_ACTION = False

ARM_JOINTS = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]

# Top-down grasp configuration; only arm_lift changes during an episode.
ARM_FLEX = -math.pi / 2
WRIST_FLEX = -math.pi / 2
GRIPPER_OPEN = 1.0
TARGET_MODEL = "target_object"
TABLE_MODEL = "pick_table"
# hand_palm_link is a massless frame folded away by the SDF conversion; this link
# sits at the same origin and exists in Gazebo (see GRASP_FIX_NOTE).
GRASP_FIX_LINK = ("hsrb", "hand_motor_dummy_link")

# hand_motor_joint -> finger tip separation [m], measured in Fortress:
#   1.20 -> 0.1349, 0.60 -> 0.0834, 0.00 -> 0.0044
_TIP_SEP_AT_OPEN = 0.1349
_TIP_SEP_AT_CLOSED = 0.0044
_MOTOR_AT_OPEN = 1.20


def motor_for_separation(separation: float) -> float:
    """Inverse of the (near linear) hand_motor_joint -> finger separation map."""
    span = _TIP_SEP_AT_OPEN - _TIP_SEP_AT_CLOSED
    return float(np.clip(_MOTOR_AT_OPEN * (separation - _TIP_SEP_AT_CLOSED) / span, 0.0, 1.2))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class PickTask(Node):
    # episode phases
    RESET = "reset"
    APPROACH = "approach"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    WATCH = "watch"
    JUDGE = "judge"
    DONE = "done"

    def __init__(self) -> None:
        super().__init__("hsr_pick_task")

        p = self.declare_parameter
        p("update_freq", 10.0)
        p("num_episodes", 10)
        p("seed", 0)
        p("world", "default")
        # table (a solid counter; the base cannot drive underneath it)
        p("table_near_edge_x", 1.55)
        p("table_depth", 0.35)
        p("table_width", 1.8)
        p("table_height", 0.42)
        # The pick_table world already contains the table; only spawn one when
        # running in a world that does not have it.
        p("spawn_table", False)
        # object placement band, measured from the near edge
        p("object_margin_min", 0.03)
        p("object_margin_max", 0.19)
        p("object_y_range", 0.55)
        p("objects", [o.name for o in OBJECT_LIBRARY])
        # randomised start pose, expressed relative to the grasp pose
        p("start_distance_min", 0.35)
        p("start_distance_max", 0.75)
        p("start_lateral", 0.30)
        p("start_yaw_jitter", 0.22)
        # motion
        p("head_tilt", -0.45)
        p("approach_timeout", 8.0)
        p("position_tolerance", 0.012)
        p("yaw_tolerance", 0.030)
        p("base_kp_linear", 1.3)
        p("base_kp_angular", 1.8)
        p("base_max_linear", 0.30)
        p("base_max_angular", 0.70)
        p("pre_grasp_height", 0.16)
        p("lift_height", 0.20)
        p("grasp_squeeze", 0.012)
        # The Ignition gripper only holds an object when the grasping flag is set
        # through the grasp action; a plain position command just moves the
        # fingers. Mirror the ROS 1 node's "hybrid" gripper mode: publish a
        # continuous opening command (which is what ends up in the recorded
        # action) and fire the grasp action once it crosses the threshold, which
        # is exactly what hsr_openpi_node does with gripper_mode:=hybrid.
        # "continuous": close with a width matched position command only
        # "hybrid"    : position command down to the threshold, then grasp action
        p("gripper_mode", "continuous")
        p("gripper_close_value", 0.10)
        p("gripper_close_threshold", 0.25)
        p("grasp_effort", -0.30)
        # Geometric grasp condition evaluated when the gripper finishes closing.
        # This *is* the task metric: the policy has to put the finger tips around
        # the object. See GRASP_FIX_NOTE for why a weld is needed at all.
        p("grasp_xy_tolerance", 0.045)
        p("grasp_z_tolerance", 0.030)
        p("grasp_fix", True)
        # Require the gripper to actually be closing before a grasp counts, so a
        # policy cannot "succeed" by hovering over the object with an open hand.
        p("grasp_gripper_max", 0.80)
        # "script"   : this node performs the pick (demonstration collection)
        # "external" : something else drives the robot (policy evaluation); this
        #              node only resets the scene, watches for the grasp
        #              condition and scores the episode.
        p("drive", "script")
        p("episode_timeout", 25.0)
        p("settle_time", 1.6)
        p("descend_time", 1.8)
        p("close_time", 1.4)
        p("lift_time", 1.8)
        p("judge_time", 0.8)
        p("success_z_margin", 0.05)
        p("results_path", "")
        p("auto_start", True)
        # topics
        p("arm_command_topic", "/arm_trajectory_controller/joint_trajectory")
        p("head_command_topic", "/head_trajectory_controller/joint_trajectory")
        p("gripper_command_topic", "/gripper_controller/joint_trajectory")
        p("base_command_topic", "/omni_base_controller/cmd_vel")
        p("joint_states_topic", "/joint_states")
        p("odom_topic", "/odom")
        p("object_pose_topic", "/gz/dynamic_pose")
        p("control_mode_topic", "/control_mode")
        p("episode_topic", "~/episode")
        # The per-episode task string is also published as a plain instruction so
        # hsr_openpi_node picks it up on its ~/instruction topic during evaluation.
        p("instruction_topic", "/hsr_openpi/instruction")

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.update_freq = float(g("update_freq"))
        self.dt = 1.0 / self.update_freq
        self.num_episodes = int(g("num_episodes"))
        self.rng = np.random.default_rng(int(g("seed")))
        self.head_tilt = float(g("head_tilt"))

        self.table_near_edge_x = float(g("table_near_edge_x"))
        self.table_depth = float(g("table_depth"))
        self.table_width = float(g("table_width"))
        self.table_height = float(g("table_height"))

        self.objects: List[ObjectSpec] = [o for o in OBJECT_LIBRARY if o.name in set(g("objects"))]
        if not self.objects:
            raise RuntimeError("no objects selected")

        self.world = GzWorld(world=str(g("world")))

        # -- publishers ---------------------------------------------------- #
        self.arm_pub = self.create_publisher(JointTrajectory, g("arm_command_topic"), command_qos())
        self.head_pub = self.create_publisher(JointTrajectory, g("head_command_topic"), command_qos())
        self.gripper_pub = self.create_publisher(JointTrajectory, g("gripper_command_topic"), command_qos())
        self.base_pub = self.create_publisher(Twist, g("base_command_topic"), command_qos())
        self.control_mode_pub = self.create_publisher(String, g("control_mode_topic"), command_qos(10))
        self.episode_pub = self.create_publisher(String, str(g("episode_topic")), command_qos(10))
        self.instruction_pub = self.create_publisher(String, str(g("instruction_topic")), command_qos(10))

        # -- subscriptions ------------------------------------------------- #
        self._lock = threading.Lock()
        self.joint_state: Dict[str, float] = {}
        self.odom: Optional[Tuple[float, float, float]] = None
        self.object_pose: Optional[Tuple[float, float, float]] = None
        self.create_subscription(JointState, g("joint_states_topic"), self._joint_cb, sensor_qos(10))
        self.create_subscription(Odometry, g("odom_topic"), self._odom_cb, sensor_qos(10))
        self.create_subscription(TFMessage, g("object_pose_topic"), self._object_cb, sensor_qos(10))

        self.grasp_client = (
            ActionClient(self, GripperApplyEffort, "/gripper_controller/grasp")
            if _HAS_GRIPPER_ACTION
            else None
        )
        if self.grasp_client is None:
            self.get_logger().error(
                "tmc_control_msgs is unavailable: the simulated gripper cannot hold anything."
            )
        self._grasp_sent = False
        self._attached = False

        self.create_service(Trigger, "~/stop", self._stop_srv)

        # -- episode state ------------------------------------------------- #
        self.episode_index = 0
        self.phase = self.RESET
        self.phase_elapsed = 0.0
        self._phase_start: Optional[float] = None
        self.spec: Optional[ObjectSpec] = None
        self.object_xy: Tuple[float, float] = (0.0, 0.0)
        self.object_spawn_z = 0.0
        self.grasp_arm_lift = 0.0
        self.base_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.results: List[dict] = []
        self.results_path = str(g("results_path"))
        self._finished = threading.Event()
        self._episode_started = False

        self.get_logger().info(GRASP_FIX_NOTE.splitlines()[0])
        self._spawn_table()
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"hsr_pick_task: {self.num_episodes} episodes, {len(self.objects)} object types, "
            f"table at x>={self.table_near_edge_x:.2f} h={self.table_height:.2f}"
        )
        if not bool(g("auto_start")):
            self.phase = self.DONE

    # ------------------------------------------------------------------ #
    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def _joint_cb(self, msg: JointState) -> None:
        with self._lock:
            self.joint_state = {n: msg.position[i] for i, n in enumerate(msg.name)}

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        with self._lock:
            self.odom = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                yaw_from_quaternion(q.x, q.y, q.z, q.w),
            )

    def _object_cb(self, msg: TFMessage) -> None:
        for tf in msg.transforms:
            if tf.child_frame_id == TARGET_MODEL:
                t = tf.transform.translation
                with self._lock:
                    self.object_pose = (t.x, t.y, t.z)
                return

    def _stop_srv(self, request, response):
        self._finished.set()
        response.success = True
        response.message = "pick task stopped"
        return response

    # ------------------------------------------------------------------ #
    def _spawn_table(self) -> None:
        if not bool(self.get_parameter("spawn_table").value):
            self.get_logger().info("spawn_table is false: using the table baked into the world")
            return
        self.world.remove(TABLE_MODEL)
        center_x = self.table_near_edge_x + self.table_depth / 2.0
        ok = self.world.spawn(
            TABLE_MODEL,
            table_sdf(width=self.table_width, depth=self.table_depth, height=self.table_height),
            x=center_x,
            y=0.0,
            z=self.table_height / 2.0,
        )
        self.get_logger().info(f"table spawned at x={center_x:.2f}: {ok}")

    def _trajectory(self, names, values, duration_s: float) -> JointTrajectory:
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in values]
        point.time_from_start = Duration(seconds=duration_s).to_msg()
        traj.points = [point]
        return traj

    def _send_arm(self, arm_lift: float, duration_s: float) -> None:
        self.arm_pub.publish(
            self._trajectory(ARM_JOINTS, [arm_lift, ARM_FLEX, 0.0, WRIST_FLEX, 0.0], duration_s)
        )

    def _send_head(self, duration_s: float) -> None:
        self.head_pub.publish(self._trajectory(HEAD_JOINTS, [0.0, self.head_tilt], duration_s))

    def _send_gripper(self, value: float, duration_s: float = 1.0) -> None:
        """Mirror hsr_openpi_node's hybrid gripper mode.

        Below the threshold the gripper is closed through the grasp action and
        **no position command is published**: publishing one flips the hardware
        back to position drive mode, which drives the fingers straight through
        the object and ejects it (hsrb_gz_ros2_control writes the position
        command whenever drive_mode is HandPosition, overriding the force
        controlled grasp).
        """
        if str(self.get_parameter("gripper_mode").value) == "hybrid":
            threshold = float(self.get_parameter("gripper_close_threshold").value)
            if value < threshold:
                if not self._grasp_sent:
                    self._send_grasp_goal(float(self.get_parameter("grasp_effort").value))
                    self._grasp_sent = True
                return
            self._grasp_sent = False
        self.gripper_pub.publish(self._trajectory(["hand_motor_joint"], [value], duration_s))

    def _send_grasp_goal(self, effort: float) -> None:
        if self.grasp_client is None:
            return
        if not self.grasp_client.server_is_ready():
            self.grasp_client.wait_for_server(timeout_sec=1.0)
            if not self.grasp_client.server_is_ready():
                self.get_logger().warn("gripper grasp action server is not available")
                return
        goal = GripperApplyEffort.Goal()
        goal.effort = float(effort)
        self.grasp_client.send_goal_async(goal)

    def _send_base(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.angular.z = float(wz)
        self.base_pub.publish(twist)

    # ------------------------------------------------------------------ #
    def _new_episode(self) -> None:
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        rng = self.rng

        self.spec = self.objects[int(rng.integers(len(self.objects)))]
        margin = float(rng.uniform(float(g("object_margin_min")), float(g("object_margin_max"))))
        obj_x = self.table_near_edge_x + margin
        obj_y = float(rng.uniform(-float(g("object_y_range")), float(g("object_y_range"))))
        self.object_xy = (obj_x, obj_y)
        self.object_spawn_z = self.table_height + self.spec.height / 2.0

        # grasp a bit above the middle so tall objects do not topple
        grasp_z = self.table_height + min(0.6 * self.spec.height, self.spec.height - 0.02)
        self.grasp_arm_lift = float(np.clip(arm_lift_for_grasp(grasp_z), 0.0, 0.69))

        base_x, base_y = base_pose_for_grasp(obj_x, obj_y, 0.0)
        self.base_target = (base_x, base_y, 0.0)

        # start pose: behind the grasp pose, with lateral and heading jitter, so
        # the object stays inside the head camera's field of view.
        back = float(rng.uniform(float(g("start_distance_min")), float(g("start_distance_max"))))
        lateral = float(rng.uniform(-float(g("start_lateral")), float(g("start_lateral"))))
        yaw = float(rng.uniform(-float(g("start_yaw_jitter")), float(g("start_yaw_jitter"))))
        start_x = base_x - back
        start_y = base_y + lateral

        if self._attached:
            self.world.publish_empty(f"/{TARGET_MODEL}/detach")
        self._attached = False
        self._grasp_info = {}
        self.world.remove(TARGET_MODEL)
        self.world.set_pose("hsrb", x=start_x, y=start_y, z=0.0, yaw=yaw)
        self._send_base()
        # Start every episode from the same arm configuration as the
        # demonstrations, whoever drives afterwards.
        self._send_arm(self.grasp_arm_lift + float(g("pre_grasp_height")), 1.2)
        self._send_head(1.0)
        self._send_gripper(GRIPPER_OPEN, 1.0)
        self.world.spawn(
            TARGET_MODEL,
            self.spec.to_sdf(TARGET_MODEL),
            x=obj_x,
            y=obj_y,
            z=self.object_spawn_z + 0.005,
            yaw=float(rng.uniform(-math.pi, math.pi)),
        )
        with self._lock:
            self.object_pose = None
        self._episode_started = False
        self.instruction_pub.publish(String(data=self._task_string()))
        self.get_logger().info(
            f"episode {self.episode_index}: {self.spec.name} at ({obj_x:.2f}, {obj_y:.2f}), "
            f"start ({start_x:.2f}, {start_y:.2f}, {yaw:+.2f}), arm_lift={self.grasp_arm_lift:.3f}"
        )

    def _task_string(self) -> str:
        return f"pick up the {self.spec.name.replace('_', ' ')}" if self.spec else "pick up the object"

    # ------------------------------------------------------------------ #
    def _base_control(self) -> bool:
        """Drive the base toward the grasp pose. Returns True when converged."""
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        with self._lock:
            odom = self.odom
        if odom is None:
            return False
        x, y, yaw = odom
        tx, ty, tyaw = self.base_target
        ex, ey = tx - x, ty - y
        eyaw = wrap_angle(tyaw - yaw)

        # rotate the world-frame error into the base frame
        c, s = math.cos(-yaw), math.sin(-yaw)
        bx = c * ex - s * ey
        by = s * ex + c * ey

        dist = math.hypot(ex, ey)
        if dist < float(g("position_tolerance")) and abs(eyaw) < float(g("yaw_tolerance")):
            self._send_base()
            return True

        kp_lin = float(g("base_kp_linear"))
        kp_ang = float(g("base_kp_angular"))
        vmax = float(g("base_max_linear"))
        wmax = float(g("base_max_angular"))
        vx = float(np.clip(kp_lin * bx, -vmax, vmax))
        vy = float(np.clip(kp_lin * by, -vmax, vmax))
        wz = float(np.clip(kp_ang * eyaw, -wmax, wmax))
        self._send_base(vx, vy, wz)
        return False

    def _gripper_grasp_signal(self) -> Tuple[bool, float, float]:
        """Real-robot-style grasp signal: the motor stalls and the springs deflect."""
        with self._lock:
            js = dict(self.joint_state)
        motor = js.get("hand_motor_joint", 0.0)
        spring = abs(js.get("hand_l_spring_proximal_joint", 0.0)) + abs(
            js.get("hand_r_spring_proximal_joint", 0.0)
        )
        target = (
            self._close_target()
            if str(self.get_parameter("gripper_mode").value) == "continuous"
            else float(self.get_parameter("gripper_close_value").value)
        )
        stalled = motor > target + 0.03
        return bool(stalled and spring > 0.02), motor, spring

    def _close_target(self) -> float:
        squeeze = float(self.get_parameter("grasp_squeeze").value)
        width = self.spec.width if self.spec else 0.05
        return motor_for_separation(max(width - squeeze, 0.005))

    # -- grasp fix -------------------------------------------------------- #
    def _palm_world(self) -> Optional[Tuple[float, float, float]]:
        """Palm position in world coordinates, from odom and the fixed arm pose."""
        with self._lock:
            odom = self.odom
            js = dict(self.joint_state)
        if odom is None or "arm_lift_joint" not in js:
            return None
        x, y, yaw = odom
        c, s_ = math.cos(yaw), math.sin(yaw)
        px = x + PALM_OFFSET_X * c - PALM_OFFSET_Y * s_
        py = y + PALM_OFFSET_X * s_ + PALM_OFFSET_Y * c
        pz = js["arm_lift_joint"] + 0.194
        return px, py, pz

    def grasp_condition(self) -> Tuple[bool, dict]:
        """True when the finger tips are around the object.

        Horizontal: the palm axis is within ``grasp_xy_tolerance`` of the object.
        Vertical:   the finger tips are inside the object's height, with
                    ``grasp_z_tolerance`` of slack at either end.
        """
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        palm = self._palm_world()
        with self._lock:
            pose = self.object_pose
        if palm is None or pose is None or self.spec is None:
            return False, {"reason": "missing pose"}
        dxy = math.hypot(palm[0] - pose[0], palm[1] - pose[1])
        tip_z = palm[2] - FINGER_TIP_BELOW_PALM_OPEN
        half = self.spec.height / 2.0
        tol = float(g("grasp_z_tolerance"))
        inside = (pose[2] - half - tol) <= tip_z <= (pose[2] + half + tol)
        with self._lock:
            motor = self.joint_state.get("hand_motor_joint", 1.2)
        closing = motor <= float(g("grasp_gripper_max"))
        ok = bool(dxy <= float(g("grasp_xy_tolerance")) and inside and closing)
        return ok, {
            "dxy": round(dxy, 4),
            "tip_z": round(tip_z, 4),
            "object_z": round(pose[2], 4),
            "vertical_ok": bool(inside),
            "hand_motor": round(motor, 3),
            "closing": bool(closing),
        }

    def _attach_object(self) -> bool:
        """Weld the object to the hand by respawning it with a DetachableJoint."""
        with self._lock:
            pose = self.object_pose
        if pose is None or self.spec is None:
            return False
        self.world.remove(TARGET_MODEL)
        ok = self.world.spawn(
            TARGET_MODEL,
            self.spec.to_sdf(TARGET_MODEL, attach_to=GRASP_FIX_LINK),
            x=pose[0],
            y=pose[1],
            z=pose[2],
        )
        self._attached = bool(ok)
        return self._attached

    def _judge(self) -> dict:
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        with self._lock:
            pose = self.object_pose
        lifted_z = pose[2] if pose else float("nan")
        ground_truth = bool(pose is not None and lifted_z > self.object_spawn_z + float(g("success_z_margin")))
        gripper_ok, motor, spring = self._gripper_grasp_signal()
        return {
            "episode": self.episode_index,
            "object": self.spec.name if self.spec else "",
            "object_xy": [round(v, 4) for v in self.object_xy],
            "spawn_z": round(self.object_spawn_z, 4),
            "final_z": None if pose is None else round(lifted_z, 4),
            "success": ground_truth,
            "gripper_success": gripper_ok,
            "grasp_condition": getattr(self, "_grasp_info", {}),
            "attached": self._attached,
            "hand_motor": round(motor, 4),
            "spring_deflection": round(spring, 4),
        }

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if self.phase == self.DONE:
            return
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        # Recording puts enough load on the box that the timer cannot keep its
        # nominal rate; measuring the phase against the clock keeps the episode
        # structure identical whether or not a bag is being written.
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._phase_start is None:
            self._phase_start = now
        self.phase_elapsed = max(now - self._phase_start, 0.0)

        if self.phase == self.RESET:
            self.control_mode_pub.publish(String(data="reset"))
            if self.phase_elapsed <= self.dt * 1.5:
                self._new_episode()
            if self.phase_elapsed >= float(g("settle_time")):
                self.phase = self.WATCH if str(g("drive")) == "external" else self.APPROACH
                self.phase_elapsed = 0.0
                self._phase_start = None
                task = self._task_string()
                self.episode_pub.publish(String(data=f"start {self.episode_index} {task}"))
                self.instruction_pub.publish(String(data=task))
                self._episode_started = True
            return

        self.control_mode_pub.publish(String(data="auto"))

        if self.phase == self.WATCH:
            # Somebody else is driving; only watch for the grasp and score it.
            if not self._attached:
                ok, info = self.grasp_condition()
                self._grasp_info = info
                if ok and bool(g("grasp_fix")):
                    self._attach_object()
                    self.get_logger().info(f"grasp condition met {info} -> attached={self._attached}")
            if self.phase_elapsed >= float(g("episode_timeout")):
                self.phase = self.JUDGE
                self.phase_elapsed = 0.0
                self._phase_start = None
            return

        if self.phase == self.APPROACH:
            self._send_arm(self.grasp_arm_lift + float(g("pre_grasp_height")), self.dt * 2)
            self._send_head(self.dt * 2)
            self._send_gripper(GRIPPER_OPEN)
            converged = self._base_control()
            if converged or self.phase_elapsed >= float(g("approach_timeout")):
                self._send_base()
                self.phase = self.DESCEND
                self.phase_elapsed = 0.0
                self._phase_start = None
            return

        self._send_base()  # the base stays still for the rest of the episode
        self._send_head(self.dt * 2)

        if self.phase == self.DESCEND:
            self._send_arm(self.grasp_arm_lift, max(float(g("descend_time")) - self.phase_elapsed, self.dt))
            self._send_gripper(GRIPPER_OPEN)
            if self.phase_elapsed >= float(g("descend_time")):
                self.phase = self.CLOSE
                self.phase_elapsed = 0.0
                self._phase_start = None
            return

        if self.phase == self.CLOSE:
            self._send_arm(self.grasp_arm_lift, self.dt * 2)
            close_time = float(g("close_time"))
            closed = (
                self._close_target()
                if str(g("gripper_mode")) == "continuous"
                else float(g("gripper_close_value"))
            )
            alpha = float(np.clip(self.phase_elapsed / max(close_time * 0.6, 1e-3), 0.0, 1.0))
            self._send_gripper(GRIPPER_OPEN + alpha * (closed - GRIPPER_OPEN), self.dt * 2)
            if self.phase_elapsed >= close_time and not self._attached:
                ok, info = self.grasp_condition()
                self._grasp_info = info
                if ok and bool(g("grasp_fix")):
                    self._attach_object()
                    self.get_logger().info(f"grasp condition met {info} -> attached={self._attached}")
                else:
                    self.get_logger().info(f"grasp condition NOT met {info}")
            if self.phase_elapsed >= close_time:
                self.phase = self.LIFT
                self.phase_elapsed = 0.0
                self._phase_start = None
            return

        if self.phase == self.LIFT:
            target = min(self.grasp_arm_lift + float(g("lift_height")), 0.69)
            self._send_arm(target, max(float(g("lift_time")) - self.phase_elapsed, self.dt))
            hold = (
                self._close_target()
                if str(g("gripper_mode")) == "continuous"
                else float(g("gripper_close_value"))
            )
            self._send_gripper(hold, self.dt * 2)
            if self.phase_elapsed >= float(g("lift_time")):
                self.phase = self.JUDGE
                self.phase_elapsed = 0.0
                self._phase_start = None
            return

        if self.phase == self.JUDGE:
            if self.phase_elapsed < float(g("judge_time")):
                return
            result = self._judge()
            self.results.append(result)
            n_ok = sum(1 for r in self.results if r["success"])
            self.get_logger().info(
                f"episode {self.episode_index}: {result['object']} "
                f"success={result['success']} (gripper={result['gripper_success']}) "
                f"z {result['spawn_z']:.3f} -> {result['final_z']} | "
                f"running {n_ok}/{len(self.results)} = {100.0 * n_ok / len(self.results):.1f}%"
            )
            if self._episode_started:
                self.episode_pub.publish(String(data=f"end {self.episode_index}"))
            self._write_results()
            self.episode_index += 1
            if self.num_episodes and self.episode_index >= self.num_episodes:
                self.control_mode_pub.publish(String(data="stopped"))
                self.episode_pub.publish(String(data="done"))
                self.world.remove(TARGET_MODEL)
                self.phase = self.DONE
                self._finished.set()
                return
            self.phase = self.RESET
            self.phase_elapsed = 0.0
            self._phase_start = None

    def _write_results(self) -> None:
        if not self.results_path:
            return
        path = pathlib.Path(self.results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_ok = sum(1 for r in self.results if r["success"])
        payload = {
            "episodes": len(self.results),
            "success": n_ok,
            "success_rate": n_ok / max(len(self.results), 1),
            "per_object": {},
            "results": self.results,
        }
        for r in self.results:
            entry = payload["per_object"].setdefault(r["object"], {"n": 0, "ok": 0})
            entry["n"] += 1
            entry["ok"] += int(r["success"])
        path.write_text(json.dumps(payload, indent=2))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickTask()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._send_base()
            node._write_results()
        except Exception:
            pass
        time.sleep(0.5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
