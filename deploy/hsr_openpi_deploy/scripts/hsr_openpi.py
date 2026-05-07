#!/usr/bin/env python3
from collections import deque
import pathlib

import os
import re
import time
from typing import Any, Optional

from actionlib import SimpleActionClient
import cv2
from geometry_msgs.msg import Twist
from hsr_data_msgs.srv import StringTrigger
from hsr_data_msgs.srv import StringTriggerResponse
import numpy as np

# ros関連
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import String 
from tmc_control_msgs.msg import GripperApplyEffortAction
from tmc_control_msgs.msg import GripperApplyEffortActionGoal
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from openpi.policies import policy_config

# openpi関連
from openpi.training import config
from openpi.training import experiment_config


MODE_CONTINUOUS = "continuous"
MODE_DISCRETE = "discrete"
MODE_HYBRID = "hybrid"
MODES = [MODE_CONTINUOUS, MODE_DISCRETE, MODE_HYBRID]

UPSAMPLE_METHOD_SPLINE = "spline"
UPSAMPLE_METHOD_LINEAR = "linear"
UPSAMPLE_METHODS = [UPSAMPLE_METHOD_SPLINE, UPSAMPLE_METHOD_LINEAR]

ACTION_SMOOTHING_NONE = "none"
ACTION_SMOOTHING_EMA = "ema"
ACTION_SMOOTHING_MA = "moving_average"
ACTION_SMOOTHING_METHODS = [ACTION_SMOOTHING_NONE, ACTION_SMOOTHING_EMA, ACTION_SMOOTHING_MA]


def _float_tag(x: float, *, ndigits: int = 3) -> str:
    """Filesystem-friendly float tag, e.g. 0.2 -> 0p200, -1.5 -> m1p500."""
    try:
        x = float(x)
    except Exception:
        return "nan"
    sign = "m" if x < 0 else ""
    x = abs(x)
    s = f"{x:.{ndigits}f}".replace(".", "p")
    return f"{sign}{s}"


def _build_trace_group_name(
    *,
    config_name: str,
    adopted_action_chunks: int,
    update_freq: int,
    upsample: bool,
    upsample_hz: int,
    upsample_method: str,
    action_smoothing: str,
    ema_alpha: float,
    ma_window: int,
    smooth_gripper: bool,
    smooth_base: bool,
) -> str:
    parts: list[str] = []
    parts.append(f"ac{int(adopted_action_chunks)}")
    parts.append(f"uf{int(update_freq)}")

    if upsample:
        parts.append(f"up{int(upsample_hz)}")
        parts.append(f"um{str(upsample_method)}")
    else:
        parts.append("original")

    if action_smoothing and action_smoothing != ACTION_SMOOTHING_NONE:
        parts.append(f"sm{str(action_smoothing)}")
        if action_smoothing == ACTION_SMOOTHING_EMA:
            parts.append(f"a{_float_tag(ema_alpha)}")
        elif action_smoothing == ACTION_SMOOTHING_MA:
            parts.append(f"w{int(ma_window)}")
        if smooth_gripper:
            parts.append("sg")
        if smooth_base:
            parts.append("sb")

    return f"{config_name}_" + "_".join(parts)


def _natural_cubic_spline_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Natural cubic spline interpolation (2nd derivative = 0 at both ends).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Strictly increasing knot positions.
    y : np.ndarray, shape (N, D)
        Values at knots.
    xq : np.ndarray, shape (M,)
        Query positions in [x[0], x[-1]].

    Returns
    -------
    np.ndarray, shape (M, D)
        Interpolated values.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    xq = np.asarray(xq, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("y must have the same length as x.")
    if x.shape[0] < 2:
        return np.repeat(y[:1], xq.shape[0], axis=0)

    n = x.shape[0]
    dim = y.shape[1]
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")

    if n == 2:
        # Linear interpolation fallback.
        t = (xq - x[0]) / (x[1] - x[0])
        t = t[:, None]
        return y[0:1] * (1.0 - t) + y[1:2] * t

    # Solve for second-derivative coefficients c (natural boundary: c0=cn-1=0).
    c = np.zeros((n, dim), dtype=np.float32)
    m = n - 2  # number of interior points
    if m > 0:
        lower = h[:-1]  # (m-1,)
        diag = 2.0 * (h[:-1] + h[1:])  # (m,)
        upper = h[1:]  # (m-1,)
        rhs = 3.0 * (
            (y[2:] - y[1:-1]) / h[1:, None] - (y[1:-1] - y[:-2]) / h[:-1, None]
        )  # (m, dim)

        if m == 1:
            c[1:-1] = rhs / diag[:, None]
        else:
            # Thomas algorithm for tridiagonal systems.
            cp = np.empty((m - 1,), dtype=np.float32)
            dp = np.empty((m, dim), dtype=np.float32)

            cp[0] = upper[0] / diag[0]
            dp[0] = rhs[0] / diag[0]
            for i in range(1, m - 1):
                denom = diag[i] - lower[i - 1] * cp[i - 1]
                cp[i] = upper[i] / denom
                dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / denom
            denom = diag[m - 1] - lower[m - 2] * cp[m - 2]
            dp[m - 1] = (rhs[m - 1] - lower[m - 2] * dp[m - 2]) / denom

            c_inner = np.empty((m, dim), dtype=np.float32)
            c_inner[m - 1] = dp[m - 1]
            for i in range(m - 2, -1, -1):
                c_inner[i] = dp[i] - cp[i] * c_inner[i + 1]
            c[1:-1] = c_inner

    # Coefficients for each segment [x_i, x_{i+1})
    b = (y[1:] - y[:-1]) / h[:, None] - (h[:, None] * (2.0 * c[:-1] + c[1:]) / 3.0)
    d = (c[1:] - c[:-1]) / (3.0 * h[:, None])
    a = y[:-1]
    c_seg = c[:-1]

    # Evaluate.
    idx = np.searchsorted(x[1:], xq, side="right")
    idx = np.clip(idx, 0, n - 2)
    dx = (xq - x[idx])[:, None]
    return a[idx] + b[idx] * dx + c_seg[idx] * (dx**2) + d[idx] * (dx**3)


def _linear_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Piecewise linear interpolation.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Strictly increasing knot positions.
    y : np.ndarray, shape (N, D)
        Values at knots.
    xq : np.ndarray, shape (M,)
        Query positions in [x[0], x[-1]].

    Returns
    -------
    np.ndarray, shape (M, D)
        Interpolated values.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    xq = np.asarray(xq, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("y must have the same length as x.")
    if x.shape[0] < 2:
        return np.repeat(y[:1], xq.shape[0], axis=0)

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")

    # Find segment indices so that x[idx] <= xq < x[idx+1].
    idx = np.searchsorted(x[1:], xq, side="right")
    idx = np.clip(idx, 0, x.shape[0] - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    denom = (x1 - x0)[:, None]
    # Avoid division by zero in pathological cases.
    denom = np.where(denom == 0, 1.0, denom)
    w = ((xq - x0)[:, None]) / denom
    return (y0 * (1.0 - w) + y1 * w).astype(np.float32, copy=False)


def _cubic_spline_upsample_actions(
    actions: np.ndarray,
    *,
    in_hz: float,
    out_hz: float,
    out_steps: Optional[int] = None,
) -> np.ndarray:
    """Upsample action sequence with natural cubic spline interpolation."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must be 2D (T, D).")
    in_steps = int(actions.shape[0])
    if out_steps is None:
        out_steps = int(round(in_steps * float(out_hz) / float(in_hz)))
    out_steps = max(int(out_steps), 1)

    if in_steps == 0:
        return actions
    if in_steps == 1:
        return np.repeat(actions, out_steps, axis=0)
    if out_steps == in_steps:
        return actions

    duration_s = in_steps / float(in_hz)
    x = np.linspace(0.0, duration_s, in_steps + 1, dtype=np.float32)
    y = np.concatenate([actions, actions[-1:, :]], axis=0)
    xq = np.arange(out_steps, dtype=np.float32) / float(out_hz)
    return _natural_cubic_spline_interpolate(x, y, xq).astype(np.float32, copy=False)


def _linear_upsample_actions(
    actions: np.ndarray,
    *,
    in_hz: float,
    out_hz: float,
    out_steps: Optional[int] = None,
) -> np.ndarray:
    """Upsample action sequence with linear interpolation."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must be 2D (T, D).")
    in_steps = int(actions.shape[0])
    if out_steps is None:
        out_steps = int(round(in_steps * float(out_hz) / float(in_hz)))
    out_steps = max(int(out_steps), 1)

    if in_steps == 0:
        return actions
    if in_steps == 1:
        return np.repeat(actions, out_steps, axis=0)
    if out_steps == in_steps:
        return actions

    duration_s = in_steps / float(in_hz)
    x = np.linspace(0.0, duration_s, in_steps + 1, dtype=np.float32)
    y = np.concatenate([actions, actions[-1:, :]], axis=0)
    xq = np.arange(out_steps, dtype=np.float32) / float(out_hz)
    return _linear_interpolate(x, y, xq).astype(np.float32, copy=False)


class ActionSmoother:
    def __init__(
        self,
        *,
        method: str,
        ema_alpha: float,
        ma_window: int,
        dims_mask: np.ndarray,
    ):
        self.method = str(method)
        if self.method not in ACTION_SMOOTHING_METHODS:
            rospy.logwarn(
                "Unknown action_smoothing '%s'. Falling back to '%s'. Available: %s",
                self.method,
                ACTION_SMOOTHING_NONE,
                ", ".join(ACTION_SMOOTHING_METHODS),
            )
            self.method = ACTION_SMOOTHING_NONE

        self.ema_alpha = float(ema_alpha)
        self.ema_alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
        self.ma_window = max(int(ma_window), 1)
        self.dims_mask = np.asarray(dims_mask, dtype=bool).reshape(-1)

        self._ema_state: Optional[np.ndarray] = None
        self._ma_buf: deque[np.ndarray] = deque(maxlen=self.ma_window)

    def update(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.method == ACTION_SMOOTHING_NONE:
            return action

        if self.dims_mask.shape[0] != action.shape[0]:
            # If shape mismatch, fall back to smoothing all dims.
            mask = np.ones_like(action, dtype=bool)
        else:
            mask = self.dims_mask

        out = np.array(action, copy=True)
        if self.method == ACTION_SMOOTHING_EMA:
            if self._ema_state is None or self._ema_state.shape != action.shape:
                self._ema_state = np.array(action, copy=True)
                return out
            a = self.ema_alpha
            self._ema_state[mask] = a * action[mask] + (1.0 - a) * self._ema_state[mask]
            out[mask] = self._ema_state[mask]
            return out

        # Moving average
        self._ma_buf.append(action)
        if len(self._ma_buf) == 0:
            return out
        stacked = np.stack(list(self._ma_buf), axis=0)
        out[mask] = stacked[:, mask].mean(axis=0).astype(np.float32, copy=False)
        return out


def _available_openpi_config_names() -> list[str]:
    configs_dict = getattr(config, "_CONFIGS_DICT", None)
    if isinstance(configs_dict, dict):
        return sorted(configs_dict.keys())
    return []


def _infer_openpi_config_name_from_checkpoint_dir(checkpoint_dir: str) -> Optional[str]:
    """
    Heuristic fallback: infer an OpenPI config name by searching the checkpoint path.

    This helps when users accidentally pass an experiment/run name as `config_name`.
    """
    names = _available_openpi_config_names()
    if not names:
        return None

    path = str(checkpoint_dir)
    matches = [name for name in names if name in path]
    matches = sorted(set(matches), key=len, reverse=True)
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_openpi_train_config(config_name: str, checkpoint_dir: str) -> config.TrainConfig:
    try:
        return config.get_config(config_name)
    except ValueError as e:
        inferred = _infer_openpi_config_name_from_checkpoint_dir(checkpoint_dir)
        if inferred is not None:
            rospy.logwarn(
                "Config '%s' not found; inferred '%s' from checkpoint_dir '%s'.",
                config_name,
                inferred,
                checkpoint_dir,
            )
            return config.get_config(inferred)

        available = _available_openpi_config_names()
        hint = ""
        if available:
            hint = f" Available configs: {', '.join(available)}"
        raise ValueError(str(e) + hint) from e


class HSREnv:
    """
    ROS経由でHSRロボットのセンサ情報の取得やアクションの実行を行う環境クラス.
    """

    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    # GRIPPER_CLOSE_THRESHOLD = 0.5  # グリッパーを閉じる閾値
    GRIPPER_CLOSE_THRESHOLD = 0.25  # グリッパーを閉じる閾値

    def __init__(self, update_freq=10):
        self.update_freq = update_freq
        self.rate = rospy.Rate(self.update_freq)

        # センサ情報の初期化
        self.head_rgb = None
        self.hand_rgb = None
        self.joint_state = None
        self.gripper_state = 0
        self.control_mode = None
        self.gripper_mode = rospy.get_param("~gripper_mode", "continuous")
        self.instruction = rospy.get_param("~instruction", "Grasp the apple.")

        self.joint_state_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]

        self.arm_action_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        self.head_action_names: list[str] = ["head_pan_joint", "head_tilt_joint"]
        self.base_action_names: list[str] = ["base_x", "base_y", "base_theta"]

        # パブリッシャーの初期化
        self.arm_pub = rospy.Publisher("/hsrb/arm_trajectory_controller/command", JointTrajectory, queue_size=1)
        self.head_pub = rospy.Publisher("/hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=1)
        self.gripper_pub = rospy.Publisher("/hsrb/gripper_controller/command", JointTrajectory, queue_size=1)
        self.base_pub = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=1)
        self.gripper_close_client = SimpleActionClient("/hsrb/gripper_controller/grasp", GripperApplyEffortAction)

        # サービス登録 (instruction 更新用)
        rospy.Service("/hsr_openpi/update_instruction", StringTrigger, self.update_instruction_srv)

        # サブスクライバーの初期化
        rospy.Subscriber(
            "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed",
            CompressedImage,
            self.head_image_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/hsrb/hand_camera/image_raw/compressed", CompressedImage, self.hand_image_callback, queue_size=1
        )
        rospy.Subscriber("/hsrb/joint_states", JointState, self.joint_state_callback, queue_size=1)
        rospy.Subscriber("/hsrb/gripper_controller/command", JointTrajectory, self.gripper_open_callback, queue_size=1)
        rospy.Subscriber(
            "/hsrb/gripper_controller/grasp/goal",
            GripperApplyEffortActionGoal,
            self.gripper_close_callback,
            queue_size=1,
        )
        rospy.Subscriber("/control_mode", String, self.control_mode_callback, queue_size=1)

    def head_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
        self.head_rgb = np.array(image)

    def hand_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
        self.hand_rgb = np.array(image)

    def joint_state_callback(self, msg: JointState):
        joints = [msg.position[msg.name.index(name)] for name in self.joint_state_names]
        self.joint_state = np.asarray(joints, dtype=np.float32)

    def gripper_open_callback(self, msg: JointTrajectory):
        self.gripper_state = self.GRIPPER_OPEN

    def gripper_close_callback(self, msg: GripperApplyEffortActionGoal):
        self.gripper_state = self.GRIPPER_CLOSE

    def control_mode_callback(self, msg: String):
        self.control_mode = msg.data

    def update_instruction_srv(self, req: StringTrigger):
        self.instruction = req.message
        rospy.loginfo("Instruction updated: %s", self.instruction)
        return StringTriggerResponse(success=True)

    def reset_observation(self, *, reset_joint_state: bool = True):
        """
        センサ情報をリセットする関数.
        """
        self.head_rgb = None
        self.hand_rgb = None
        if reset_joint_state:
            self.joint_state = None

    def get_observations(self):
        """
        ロボットのセンサ情報をまとめた辞書を返す関数.
        全ての必要な情報がそろっていなければNone.
        """
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

    def execute_actions(self, action: np.ndarray) -> bool:
        """
        acitonをロボットに反映

        Parameters
        ----------
        action : np.ndarray
            ロボットに反映するアクション:
            [
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
                "base_t",
            ]

        Returns
        -------
        bool
            実行できた場合はTrue, できなかった場合はFalse
        """
        # control_modeが"auto"の場合のみ実行
        if self.control_mode != "auto":
            return False  # 実行できない場合はFalseを返す

        # アーム制御
        arm_traj = JointTrajectory()
        arm_traj.joint_names = self.arm_action_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = action[:5]
        arm_point.velocities = []
        arm_point.time_from_start = rospy.Duration(1 / self.update_freq / 2)
        arm_traj.points = [arm_point]

        # ヘッド制御
        head_traj = JointTrajectory()
        head_traj.joint_names = self.head_action_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = action[6:8]
        arm_point.velocities = []
        arm_point.time_from_start = rospy.Duration(1 / self.update_freq / 2)

        head_traj.points = [arm_point]

        # ベース制御
        twist = Twist()
        twist.linear.x = action[8]
        twist.linear.y = action[9]
        twist.angular.z = action[10]

        # グリッパー制御
        if self.gripper_mode == "continuous":
            gripper_traj = JointTrajectory()
            gripper_traj.joint_names = ["hand_motor_joint"]
            arm_point = JointTrajectoryPoint()
            gripper_value = np.clip(action[5], -0.1, 1.23)
            arm_point.positions = [gripper_value]
            arm_point.velocities = []
            arm_point.time_from_start = rospy.Duration(1)
            gripper_traj.points = [arm_point]
            self.gripper_pub.publish(gripper_traj)
        elif self.gripper_mode == "discrete":
            # グリッパーを閉じるかどうか 1: 閉じる, 0: 開く
            gripper_action = self.GRIPPER_CLOSE if action[5] < self.GRIPPER_CLOSE_THRESHOLD else self.GRIPPER_OPEN
            if self.gripper_state != gripper_action:
                if gripper_action == self.GRIPPER_CLOSE:  # グリッパーを閉じる
                    goal = GripperApplyEffortActionGoal()
                    goal.goal.effort = -0.018
                    self.gripper_close_client.send_goal(goal.goal)
                else:  # グリッパーを開く
                    arm_traj = JointTrajectory()
                    arm_traj.joint_names = ["hand_motor_joint"]
                    arm_point = JointTrajectoryPoint()
                    arm_point.positions = [1.239183768915874]
                    arm_point.velocities = []
                    arm_point.time_from_start = rospy.Duration(1)
                    arm_traj.points = [arm_point]
                    self.gripper_pub.publish(arm_traj)
                self.gripper_state = gripper_action
        elif self.gripper_mode == "hybrid":
            # hybridモード: 閉じるときはしきい値を下回るまでcontinuous、下回ったらdiscrete
            # 開くときはcontinuousと同じ
            gripper_value = action[5]
            if gripper_value < self.GRIPPER_CLOSE_THRESHOLD:
                # しきい値を下回った場合: discreteと同じ挙動（GripperApplyEffortで力を加える）
                if self.gripper_state != self.GRIPPER_CLOSE:
                    goal = GripperApplyEffortActionGoal()
                    goal.goal.effort = -0.018
                    self.gripper_close_client.send_goal(goal.goal)
                    self.gripper_state = self.GRIPPER_CLOSE
            else:
                # しきい値以上の場合: continuousと同じ挙動
                gripper_traj = JointTrajectory()
                gripper_traj.joint_names = ["hand_motor_joint"]
                arm_point = JointTrajectoryPoint()
                gripper_value = np.clip(gripper_value, -0.1, 1.23)
                arm_point.positions = [gripper_value]
                arm_point.velocities = []
                arm_point.time_from_start = rospy.Duration(1)
                gripper_traj.points = [arm_point]
                self.gripper_pub.publish(gripper_traj)
                self.gripper_state = self.GRIPPER_OPEN

        self.arm_pub.publish(arm_traj)
        self.head_pub.publish(head_traj)
        self.base_pub.publish(twist)

        return True

    def sleep(self):
        self.rate.sleep()


class OpenpiPolicy:
    """
    PiZeroというpolicyの推論・実行を担当するクラスです.
    HSREnvからセンサ情報を取得し, policyの計算後にアクションを環境に反映させます.
    """

    def __init__(
        self,
        config_name: str,
        config_yaml: str,
        checkpoint_dir: str,
        adopted_action_chunks: int = 15,  # 一度の推論で得られるaction_chunkのうち、最初何個を使うか
        action_hz: int = 10,  # action_chunk の時間解像度(Hz). 通常は update_freq と同じ
        upsample: bool = False,
        upsample_hz: int = 50,
        upsample_method: str = UPSAMPLE_METHOD_SPLINE,
    ):
        # openpiのpolicyのロード
        self.config: config.TrainConfig = self._resolve_train_config(config_name, config_yaml, checkpoint_dir)
        self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir)

        self.adopted_action_chunks: int = adopted_action_chunks
        self.action_hz: int = action_hz
        self.upsample: bool = upsample
        self.upsample_hz: int = upsample_hz
        self.upsample_method: str = str(upsample_method)
        if self.upsample_method not in UPSAMPLE_METHODS:
            rospy.logwarn(
                "Unknown upsample_method '%s'. Falling back to '%s'. Available: %s",
                self.upsample_method,
                UPSAMPLE_METHOD_SPLINE,
                ", ".join(UPSAMPLE_METHODS),
            )
            self.upsample_method = UPSAMPLE_METHOD_SPLINE

        self.execution_action_chunks: int = adopted_action_chunks
        if self.upsample:
            self.execution_action_chunks = max(
                int(round(adopted_action_chunks * float(self.upsample_hz) / float(self.action_hz))),
                1,
            )

        self.action_queue: deque = deque(maxlen=self.execution_action_chunks)
        self._last_original_action_chunk: Optional[np.ndarray] = None
        self._infer_latencies_s: list[float] = []

    def _record_infer_timing(self, *, start_s: float, end_s: float) -> None:
        latency = float(end_s - start_s)
        if latency >= 0:
            self._infer_latencies_s.append(latency)

    def log_inference_stats(self) -> None:
        def _summarize(values: list[float]) -> tuple[int, float, float] | None:
            if not values:
                return None
            arr = np.asarray(values, dtype=np.float64)
            return int(arr.size), float(arr.mean()), float(arr.var())

        lat = _summarize(self._infer_latencies_s)

        if lat is None:
            rospy.loginfo("Inference stats: no inference calls recorded.")
            return

        n_lat, mean_lat, var_lat = lat
        rospy.loginfo(
            "Inference latency (infer() only): n=%d mean=%.1fms var=%.3f(ms^2)",
            n_lat,
            mean_lat * 1e3,
            var_lat * 1e6,
        )

    @staticmethod
    def _resolve_yaml_path(yaml_path: str) -> pathlib.Path:
        path = pathlib.Path(yaml_path).expanduser()
        if path.is_absolute() and path.exists():
            return path.resolve()

        repo_root = pathlib.Path(__file__).resolve().parents[3]
        candidates = [
            pathlib.Path.cwd() / path,
            repo_root / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        candidate_text = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"YAML config '{yaml_path}' not found. Tried: '{path}' (absolute), {candidate_text}"
        )

    @classmethod
    def _resolve_train_config(
        cls, config_name: str, config_yaml: str, checkpoint_dir: str
    ) -> config.TrainConfig:
        requested_yaml = config_yaml.strip()
        if not requested_yaml and config_name.lower().endswith((".yaml", ".yml")):
            requested_yaml = config_name

        if requested_yaml:
            yaml_path = cls._resolve_yaml_path(requested_yaml)
            rospy.loginfo("Load config from YAML: %s", yaml_path)
            return experiment_config.load_experiment_config(yaml_path)

        rospy.loginfo("Load config from registered config.py name: %s", config_name)
        return _resolve_openpi_train_config(config_name, checkpoint_dir)

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        センサ情報を受け取り、アクションを返す関数
        obs: Dict[str, Any]
            センサ情報
            {
                "head_rgb": <np.ndarray shape (H, W, 3)>,
                "hand_rgb": <np.ndarray shape (H, W, 3)>,
                "joint_state": <np.ndarray shape (8,)>, # ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
                "instruction": <str>,
            }
        return: np.ndarray : shape (11,)
            アクション
            [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "gripper",
                "head_pan_joint",
                "head_tilt_joint",
                "base_x",
                "base_y",
                "base_t",
            ]
        """

        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            # 差分になっている行動を元に戻す
            return action + np.concatenate(
                [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
            )  # gripper, base_x,base_y, base_t は差分を取っていないため、0
        # Policy への入力辞書を作成
        policy_input = {
            "head_rgb": obs["head_rgb"],
            "hand_rgb": obs["hand_rgb"],
            "state": obs["joint_state"],
            "prompt": obs["instruction"],
        }
        infer_start_s = time.perf_counter()
        raw_action_chunk = np.asarray(self.policy.infer(policy_input)["actions"], dtype=np.float32)
        infer_end_s = time.perf_counter()
        self._record_infer_timing(start_s=infer_start_s, end_s=infer_end_s)
        self._last_original_action_chunk = raw_action_chunk[: self.adopted_action_chunks]

        if self.upsample:
            action_chunk = raw_action_chunk[: self.adopted_action_chunks]
            if self.upsample_method == UPSAMPLE_METHOD_LINEAR:
                action_chunk = _linear_upsample_actions(
                    action_chunk,
                    in_hz=self.action_hz,
                    out_hz=self.upsample_hz,
                    out_steps=self.execution_action_chunks,
                )
            else:
                action_chunk = _cubic_spline_upsample_actions(
                    action_chunk,
                    in_hz=self.action_hz,
                    out_hz=self.upsample_hz,
                    out_steps=self.execution_action_chunks,
                )
        else:
            action_chunk = raw_action_chunk[: self.adopted_action_chunks]

        self.action_queue.extend(action_chunk[1:])
        action = action_chunk[0]  # 最初のアクションだけを返す

        # 差分になっている行動を元に戻す
        return action + np.concatenate(
            [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
        )  # gripper, base_x,base_y, base_t は差分を取っていないため、0

    def get_last_original_action_chunk(self) -> Optional[np.ndarray]:
        return self._last_original_action_chunk


class ExecTraceRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        config_name: str,
        joint_dim_names: Optional[list[str]] = None,
        base_action_names: Optional[list[str]] = None,
        base_dir: str = "/home/openpi/deploy_record",
    ):
        self.enabled = bool(enabled)
        self.config_name = str(config_name)
        self.joint_dim_names = list(joint_dim_names or [])
        self.base_action_names = list(base_action_names or [])
        self.base_dir = str(base_dir)

        self._t: list[float] = []
        self._joint_state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._t_action_original: list[float] = []
        self._action_original_delta: list[np.ndarray] = []
        self._t_chunk_start: list[float] = []

    def add(self, *, stamp_s: float, joint_state: np.ndarray, action: np.ndarray) -> None:
        if not self.enabled:
            return
        self._t.append(float(stamp_s))
        self._joint_state.append(np.asarray(joint_state, dtype=np.float32).reshape(-1))
        self._action.append(np.asarray(action, dtype=np.float32).reshape(-1))

    def add_chunk_start(self, *, stamp_s: float) -> None:
        if not self.enabled:
            return
        self._t_chunk_start.append(float(stamp_s))

    def add_original_action_chunk(self, *, base_stamp_s: float, action_chunk: np.ndarray, action_hz: float) -> None:
        if not self.enabled:
            return
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim != 2:
            return
        hz = float(action_hz)
        if hz <= 0:
            return
        dt = 1.0 / hz
        base = float(base_stamp_s)
        for k in range(int(action_chunk.shape[0])):
            self._t_action_original.append(base + k * dt)
            self._action_original_delta.append(action_chunk[k].reshape(-1))

    def _output_dir(self) -> str:
        safe_name = self.config_name.replace("/", "_").replace(os.sep, "_").strip()
        if safe_name == "":
            safe_name = "unknown_config"
        return os.path.join(self.base_dir, safe_name)

    def _next_run_index(self, out_dir: str) -> int:
        try:
            names = os.listdir(out_dir)
        except FileNotFoundError:
            return 1

        max_idx = 0
        for name in names:
            m = re.match(r"^(\d+)\.(npz|png)$", name)
            if m is None:
                continue
            try:
                idx = int(m.group(1))
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
        return max_idx + 1

    def _dim_name(self, dim_idx: int) -> str:
        if 0 <= dim_idx < len(self.joint_dim_names):
            return self.joint_dim_names[dim_idx]
        base_i = dim_idx - len(self.joint_dim_names)
        if 0 <= base_i < len(self.base_action_names):
            return self.base_action_names[base_i]
        return f"dim[{dim_idx}]"

    def save_and_plot(self) -> None:
        if not self.enabled:
            return
        if len(self._t) == 0:
            rospy.logwarn("ExecTraceRecorder: no samples to save.")
            return

        out_dir = self._output_dir()
        os.makedirs(out_dir, exist_ok=True)
        run_idx = self._next_run_index(out_dir)
        stem = f"{run_idx:04d}"

        npz_path = os.path.join(out_dir, f"{stem}.npz")
        plot_path = os.path.join(out_dir, f"{stem}.png")

        payload: dict[str, np.ndarray] = {
            "t": np.asarray(self._t, dtype=np.float64),
            "joint_state": np.stack(self._joint_state, axis=0),
            "action": np.stack(self._action, axis=0),
            "joint_dim_names": np.asarray(self.joint_dim_names, dtype=str),
            "base_action_names": np.asarray(self.base_action_names, dtype=str),
        }
        if len(self._t_chunk_start) > 0:
            payload["t_chunk_start"] = np.asarray(self._t_chunk_start, dtype=np.float64)
        if len(self._t_action_original) > 0 and len(self._action_original_delta) == len(self._t_action_original):
            payload["t_action_original"] = np.asarray(self._t_action_original, dtype=np.float64)
            payload["action_original_delta"] = np.stack(self._action_original_delta, axis=0)

        np.savez_compressed(npz_path, **payload)
        rospy.loginfo("Saved exec trace: %s", npz_path)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            rospy.logwarn("ExecTraceRecorder: matplotlib unavailable, skipping plot (%s).", e)
            return

        t = np.asarray(self._t, dtype=np.float64)
        joint_state = np.stack(self._joint_state, axis=0)
        action = np.stack(self._action, axis=0)
        t_chunk_start = np.asarray(self._t_chunk_start, dtype=np.float64) if len(self._t_chunk_start) > 0 else None
        t_action_original = (
            np.asarray(self._t_action_original, dtype=np.float64) if len(self._t_action_original) > 0 else None
        )
        action_original_delta = (
            np.stack(self._action_original_delta, axis=0) if len(self._action_original_delta) > 0 else None
        )

        n_joint = int(joint_state.shape[1]) if joint_state.ndim == 2 else 1
        n_action = int(action.shape[1]) if action.ndim == 2 else 1
        n_action_original = (
            int(action_original_delta.shape[1])
            if action_original_delta is not None and action_original_delta.ndim == 2
            else 0
        )
        nrows = max(max(n_joint, n_action), 1)

        fig_h = max(6.0, 1.1 * float(nrows))
        fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_h), sharex=True)
        if nrows == 1:
            axes = [axes]

        for i in range(nrows):
            ax = axes[i]
            has_joint = i < n_joint
            has_action = i < n_action
            has_action_original = action_original_delta is not None and i < n_action_original
            dim_name = self._dim_name(i)

            if has_joint:
                ax.plot(t, joint_state[:, i], label=f"{dim_name} (joint)")
            if has_action:
                ax.plot(t, action[:, i], label=f"{dim_name} (action)")
            if has_action_original and t_action_original is not None:
                # Convert delta->command using joint_state sampled at the closest previous time.
                idx = np.searchsorted(t, t_action_original, side="right") - 1
                idx = np.clip(idx, 0, max(len(t) - 1, 0))
                js = joint_state[idx]
                original_cmd = np.array(action_original_delta[:, i], copy=True)
                if i < 5:
                    original_cmd = original_cmd + js[:, i]
                elif 6 <= i < 8:
                    original_cmd = original_cmd + js[:, i]
                ax.plot(
                    t_action_original,
                    original_cmd,
                    linestyle="None",
                    marker="+",
                    markersize=4.5,
                    alpha=0.9,
                    label="original action",
                )
            ax.set_ylabel(dim_name)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title("joint/action (same index overlaid when available)")
            if has_joint or has_action:
                ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("time [s]")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, format="png")
        plt.close(fig)
        rospy.loginfo("Saved exec trace plot: %s", plot_path)


def main():
    print("Start hsr_openpi")

    rospy.init_node("hsr_openpi")

    config_name: str = rospy.get_param("~config_name", "pi0_hsr_low_mem_finetune")
    config_yaml: str = rospy.get_param("~config_yaml", "")
    checkpoint_dir: str = rospy.get_param(
        "~checkpoint_dir", "/home/openpi/checkpoints/pi0_hsr_low_mem_finetune/hsr_tmc_new/5000"
    )
    adopted_action_chunks = rospy.get_param("~adopted_action_chunks", 1)
    update_freq: int = rospy.get_param("~update_freq", 5)
    upsample: bool = rospy.get_param("~upsample", False)
    upsample_hz: int = rospy.get_param("~upsample_hz", 50)
    upsample_method: str = rospy.get_param("~upsample_method", UPSAMPLE_METHOD_SPLINE)
    execution_freq: int = upsample_hz if upsample else update_freq

    action_smoothing: str = rospy.get_param("~action_smoothing", ACTION_SMOOTHING_NONE)
    ema_alpha: float = rospy.get_param("~ema_alpha", 0.2)
    ma_window: int = rospy.get_param("~ma_window", 5)
    smooth_gripper: bool = rospy.get_param("~smooth_gripper", False)
    smooth_base: bool = rospy.get_param("~smooth_base", False)

    save_exec_trace: bool = rospy.get_param("~save_exec_trace", False)
    trace_group_name = _build_trace_group_name(
        config_name=config_name,
        adopted_action_chunks=int(adopted_action_chunks),
        update_freq=int(update_freq),
        upsample=bool(upsample),
        upsample_hz=int(upsample_hz),
        upsample_method=str(upsample_method),
        action_smoothing=str(action_smoothing),
        ema_alpha=float(ema_alpha),
        ma_window=int(ma_window),
        smooth_gripper=bool(smooth_gripper),
        smooth_base=bool(smooth_base),
    )

    rospy.loginfo("config_name: %s", config_name)
    rospy.loginfo("config_yaml: %s", config_yaml if config_yaml else "<not set>")
    rospy.loginfo("checkpoint_dir: %s", checkpoint_dir)
    rospy.loginfo("adopted_action_chunks: %s", adopted_action_chunks)
    rospy.loginfo("update_freq: %s", update_freq)
    rospy.loginfo("upsample: %s", upsample)
    rospy.loginfo("upsample_hz: %s", upsample_hz)
    rospy.loginfo("upsample_method: %s", upsample_method)
    rospy.loginfo("action_smoothing: %s", action_smoothing)
    rospy.loginfo("ema_alpha: %s", ema_alpha)
    rospy.loginfo("ma_window: %s", ma_window)
    rospy.loginfo("smooth_gripper: %s", smooth_gripper)
    rospy.loginfo("smooth_base: %s", smooth_base)
    rospy.loginfo("execution_freq: %s", execution_freq)
    rospy.loginfo("gripper_mode: %s", rospy.get_param("~gripper_mode", "continuous"))
    rospy.loginfo("save_exec_trace: %s", save_exec_trace)
    rospy.loginfo("exec_trace_group_name: %s", trace_group_name)

    env = HSREnv(update_freq=execution_freq)
    policy = OpenpiPolicy(
        config_name,
        config_yaml,
        checkpoint_dir,
        adopted_action_chunks,
        action_hz=update_freq,
        upsample=upsample,
        upsample_hz=upsample_hz,
        upsample_method=upsample_method,
    )
    # Default smoothing dims: arm(5) + head(2). Optionally add gripper/base.
    base_mask = np.array([True, True, True, True, True, False, True, True, False, False, False], dtype=bool)
    if smooth_gripper:
        base_mask[5] = True
    if smooth_base:
        base_mask[8:11] = True
    action_smoother = ActionSmoother(
        method=action_smoothing,
        ema_alpha=ema_alpha,
        ma_window=ma_window,
        dims_mask=base_mask,
    )
    recorder = ExecTraceRecorder(
        enabled=save_exec_trace,
        config_name=trace_group_name,
        joint_dim_names=env.joint_state_names,
        base_action_names=env.base_action_names,
    )
    rospy.on_shutdown(recorder.save_and_plot)
    rospy.on_shutdown(policy.log_inference_stats)

    log_interval = 1
    if upsample and update_freq > 0:
        log_interval = max(int(round(execution_freq / update_freq)), 1)
    tick = 0
    perf0 = time.perf_counter()
    chunk_gaps_s: list[float] = []
    last_chunk_end_t_s: Optional[float] = None

    def log_chunk_gap_stats() -> None:
        if not chunk_gaps_s:
            rospy.loginfo("Chunk gap stats: no chunk gaps recorded.")
            return
        arr = np.asarray(chunk_gaps_s, dtype=np.float64)
        rospy.loginfo(
            "Chunk gap (prev chunk last action -> next chunk first action): n=%d mean=%.1fms var=%.3f(ms^2)",
            int(arr.size),
            float(arr.mean()) * 1e3,
            float(arr.var()) * 1e6,
        )

    try:
        while not rospy.is_shutdown():
            will_infer = len(policy.action_queue) == 0
            obs = env.get_observations()
            if obs is None:
                if upsample and (not will_infer) and env.joint_state is not None:
                    obs = {"joint_state": env.joint_state, "instruction": env.instruction}
                else:
                    rospy.loginfo("Observations are not ready.")
                    tick += 1
                    env.sleep()
                    continue

            action = policy.act(obs)
            action_t_s = time.perf_counter() - perf0  # immediately after act()
            action_to_send = action_smoother.update(action)
            is_executed = env.execute_actions(action_to_send)
            sent_t_s = time.perf_counter() - perf0

            if is_executed:
                if will_infer:
                    recorder.add_chunk_start(stamp_s=sent_t_s)
                    if last_chunk_end_t_s is not None:
                        gap = float(sent_t_s - last_chunk_end_t_s)
                        if gap >= 0:
                            chunk_gaps_s.append(gap)

                # If we just consumed the last action of the current chunk, record chunk end time.
                if (not will_infer) and len(policy.action_queue) == 0:
                    last_chunk_end_t_s = sent_t_s

            if is_executed and "joint_state" in obs:
                recorder.add(
                    stamp_s=sent_t_s,
                    joint_state=obs["joint_state"],
                    action=action_to_send,
                )
                if upsample and will_infer:
                    original_chunk = policy.get_last_original_action_chunk()
                    if original_chunk is not None and original_chunk.ndim == 2:
                        recorder.add_original_action_chunk(
                            base_stamp_s=action_t_s,
                            action_chunk=original_chunk,
                            action_hz=update_freq,
                        )

            # # テストで画像を出力
            # cv2.imwrite("/root/catkin_ws/head_rgb.png", obs["head_rgb"])
            # cv2.imwrite("/root/catkin_ws/hand_rgb.png", obs["hand_rgb"])
            # # cv2.imshow("hand_rgb", obs["hand_rgb"])
            # break

            if tick % log_interval == 0:
                if is_executed:
                    rospy.loginfo("Action executed.")
                else:
                    rospy.loginfo("Action not executed.")
                rospy.loginfo("Language instruction: %s", obs.get("instruction", ""))
                rospy.loginfo("Action: %s", action)
            if not upsample:
                env.reset_observation()
            elif will_infer:
                env.reset_observation(reset_joint_state=False)
            tick += 1
            env.sleep()
    except KeyboardInterrupt:
        # Ensure recorder flush even if ROS shutdown hook doesn't run in time.
        rospy.loginfo("KeyboardInterrupt received. Saving exec trace and shutting down.")
        recorder.save_and_plot()
        policy.log_inference_stats()
        log_chunk_gap_stats()
        try:
            rospy.signal_shutdown("KeyboardInterrupt")
        except Exception:
            pass
        return


if __name__ == "__main__":
    main()
