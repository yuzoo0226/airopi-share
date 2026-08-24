#!/usr/bin/env python3
"""ROS 2 (Humble) inference node for openpi pi0 / pi0.5 policies on the HSR.

This is the ROS 2 port of ``deploy/hsr_openpi_deploy/scripts/hsr_openpi.py``.

The control loop is intentionally kept identical to the ROS 1 version: one
inference produces an action chunk, the first action is executed immediately and
the remaining ones are replayed (optionally upsampled) until the queue drains,
at which point a new inference is triggered.

Unlike the ROS 1 node, the policy itself normally lives in a separate process
(``server/serve_hsr_policy_ws.py``) because openpi requires Python 3.11 + JAX
while ROS 2 Humble ships Python 3.10. Set ``policy_backend:=local`` to run the
policy in-process instead.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from hsr_openpi.action_utils import (
    ACTION_SMOOTHING_NONE,
    UPSAMPLE_METHOD_SPLINE,
    ActionSmoother,
    build_trace_group_name,
)
from hsr_openpi.exec_trace import ExecTraceRecorder
from hsr_openpi.hsr_env import HSREnv
from hsr_openpi.openpi_policy import OpenpiPolicy
from hsr_openpi.policy_client import create_policy

try:
    from hsr_openpi_msgs.srv import StringTrigger

    _HAS_STRING_TRIGGER = True
except ImportError:  # pragma: no cover - depends on the runtime workspace
    StringTrigger = None  # type: ignore[assignment]
    _HAS_STRING_TRIGGER = False


_PARAMETERS = (
    # -- policy backend ------------------------------------------------------
    ("policy_backend", "websocket"),
    ("policy_host", "127.0.0.1"),
    ("policy_port", 8010),
    ("policy_api_key", ""),
    ("policy_connect_timeout", -1.0),
    ("config_name", ""),
    ("config_yaml", ""),
    ("checkpoint_dir", ""),
    # -- control loop --------------------------------------------------------
    ("update_freq", 10),
    ("adopted_action_chunks", 10),
    ("upsample", True),
    ("upsample_hz", 100),
    ("upsample_method", UPSAMPLE_METHOD_SPLINE),
    ("action_smoothing", "ema"),
    ("ema_alpha", 0.2),
    ("ma_window", 5),
    ("smooth_gripper", False),
    ("smooth_base", False),
    # -- robot interface -----------------------------------------------------
    ("instruction", "Grasp the bottle."),
    ("gripper_mode", "hybrid"),
    ("gripper_effort", -0.018),
    ("image_transport", "auto"),
    ("bgr_to_rgb", True),
    ("require_control_mode", True),
    ("control_mode_active_value", "auto"),
    ("auto_start", False),
    ("head_image_topic", "/head_rgbd_sensor/rgb/image_rect_color"),
    ("hand_image_topic", "/hand_camera/image_raw"),
    ("joint_states_topic", "/joint_states"),
    ("control_mode_topic", "/control_mode"),
    ("arm_command_topic", "/arm_trajectory_controller/joint_trajectory"),
    ("head_command_topic", "/head_trajectory_controller/joint_trajectory"),
    ("gripper_command_topic", "/gripper_controller/joint_trajectory"),
    ("base_command_topic", "/omni_base_controller/cmd_vel"),
    ("gripper_grasp_action", "/gripper_controller/grasp"),
    # -- diagnostics ---------------------------------------------------------
    ("save_exec_trace", False),
    ("exec_trace_dir", "/home/hsr/deploy_record"),
    ("wait_for_clock_timeout", 30.0),
)


class HsrOpenpiNode(Node):
    def __init__(self) -> None:
        super().__init__("hsr_openpi")

        for name, default in _PARAMETERS:
            self.declare_parameter(name, default)

        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.update_freq = int(g("update_freq"))
        self.adopted_action_chunks = int(g("adopted_action_chunks"))
        self.upsample = bool(g("upsample"))
        self.upsample_hz = int(g("upsample_hz"))
        self.upsample_method = str(g("upsample_method"))
        self.execution_freq = self.upsample_hz if self.upsample else self.update_freq

        self.action_smoothing = str(g("action_smoothing"))
        self.ema_alpha = float(g("ema_alpha"))
        self.ma_window = int(g("ma_window"))
        self.smooth_gripper = bool(g("smooth_gripper"))
        self.smooth_base = bool(g("smooth_base"))

        self._shutdown = threading.Event()

        # -- environment ----------------------------------------------------- #
        self.env = HSREnv(self, update_freq=self.execution_freq)
        if bool(g("auto_start")):
            self.env.set_control_mode(str(g("control_mode_active_value")))
            self.get_logger().info("auto_start is enabled: action execution starts immediately.")

        # -- policy ---------------------------------------------------------- #
        backend = str(g("policy_backend"))
        self.get_logger().info(f"Creating policy backend '{backend}' ...")
        policy_impl = create_policy(
            backend,
            host=str(g("policy_host")),
            port=int(g("policy_port")),
            api_key=str(g("policy_api_key")),
            config_name=str(g("config_name")),
            config_yaml=str(g("config_yaml")),
            checkpoint_dir=str(g("checkpoint_dir")),
            connect_timeout_s=float(g("policy_connect_timeout")),
            loginfo=self.get_logger().info,
        )
        self.policy = OpenpiPolicy(
            policy_impl,
            adopted_action_chunks=self.adopted_action_chunks,
            action_hz=self.update_freq,
            upsample=self.upsample,
            upsample_hz=self.upsample_hz,
            upsample_method=self.upsample_method,
            logwarn=self.get_logger().warn,
            loginfo=self.get_logger().info,
        )

        # -- action smoothing ------------------------------------------------ #
        dims_mask = np.array([True] * 5 + [False] + [True] * 2 + [False] * 3, dtype=bool)
        if self.smooth_gripper:
            dims_mask[5] = True
        if self.smooth_base:
            dims_mask[8:11] = True
        self.smoother = ActionSmoother(
            method=self.action_smoothing,
            ema_alpha=self.ema_alpha,
            ma_window=self.ma_window,
            dims_mask=dims_mask,
            logwarn=self.get_logger().warn,
        )

        # -- trace recorder --------------------------------------------------- #
        config_label = str(g("config_name")) or str(policy_impl.metadata.get("config_name", "openpi"))
        self.trace_group_name = build_trace_group_name(
            config_name=config_label,
            adopted_action_chunks=self.adopted_action_chunks,
            update_freq=self.update_freq,
            upsample=self.upsample,
            upsample_hz=self.upsample_hz,
            upsample_method=self.upsample_method,
            action_smoothing=self.action_smoothing,
            ema_alpha=self.ema_alpha,
            ma_window=self.ma_window,
            smooth_gripper=self.smooth_gripper,
            smooth_base=self.smooth_base,
        )
        self.recorder = ExecTraceRecorder(
            enabled=bool(g("save_exec_trace")),
            config_name=self.trace_group_name,
            joint_dim_names=self.env.JOINT_STATE_NAMES,
            base_action_names=self.env.BASE_ACTION_NAMES,
            base_dir=str(g("exec_trace_dir")),
            loginfo=self.get_logger().info,
            logwarn=self.get_logger().warn,
        )

        # -- services --------------------------------------------------------- #
        if _HAS_STRING_TRIGGER:
            self.create_service(StringTrigger, "~/update_instruction", self._update_instruction_srv)
        else:
            self.get_logger().warn("hsr_openpi_msgs is not available: ~/update_instruction is disabled.")
        self.create_service(Trigger, "~/start", self._start_srv)
        self.create_service(Trigger, "~/stop", self._stop_srv)
        self.create_subscription(String, "~/instruction", self._instruction_callback, 10)

        self._log_configuration()

    # ------------------------------------------------------------------ #
    def _log_configuration(self) -> None:
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.get_logger().info(
            "hsr_openpi configuration:\n"
            f"  policy_backend        : {g('policy_backend')}\n"
            f"  policy endpoint       : {g('policy_host')}:{g('policy_port')}\n"
            f"  config_name           : {g('config_name') or '<from server>'}\n"
            f"  checkpoint_dir        : {g('checkpoint_dir') or '<from server>'}\n"
            f"  update_freq           : {self.update_freq}\n"
            f"  adopted_action_chunks : {self.adopted_action_chunks}\n"
            f"  upsample              : {self.upsample} ({self.upsample_hz} Hz, {self.upsample_method})\n"
            f"  execution_freq        : {self.execution_freq}\n"
            f"  action_smoothing      : {self.action_smoothing} "
            f"(alpha={self.ema_alpha}, window={self.ma_window})\n"
            f"  gripper_mode          : {g('gripper_mode')}\n"
            f"  instruction           : {g('instruction')}\n"
            f"  require_control_mode  : {g('require_control_mode')} "
            f"(active value: '{g('control_mode_active_value')}')\n"
            f"  save_exec_trace       : {g('save_exec_trace')} -> {self.trace_group_name}"
        )

    # -- service / topic callbacks --------------------------------------- #
    def _update_instruction_srv(self, request, response):
        self.env.set_instruction(request.message)
        response.message = request.message
        response.success = True
        return response

    def _instruction_callback(self, msg: String) -> None:
        self.env.set_instruction(msg.data)

    def _start_srv(self, request, response):
        self.env.set_control_mode(self.get_parameter("control_mode_active_value").value)
        self.policy.reset()
        self.smoother.reset()
        response.success = True
        response.message = "action execution enabled"
        self.get_logger().info(response.message)
        return response

    def _stop_srv(self, request, response):
        self.env.set_control_mode("stopped")
        self.env.stop_base()
        response.success = True
        response.message = "action execution disabled"
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------ #
    def wait_for_clock(self) -> None:
        """With use_sim_time, rate.sleep() only advances once /clock is alive."""
        if not self.get_parameter("use_sim_time").value:
            return
        timeout = float(self.get_parameter("wait_for_clock_timeout").value)
        self.get_logger().info("use_sim_time is set: waiting for /clock ...")
        deadline = time.time() + timeout
        while rclpy.ok() and not self._shutdown.is_set():
            if self.get_clock().now().nanoseconds > 0:
                self.get_logger().info("/clock is alive.")
                return
            if time.time() > deadline:
                self.get_logger().warn(
                    f"No /clock message after {timeout:.0f}s. Is Gazebo running with the ros_gz clock bridge?"
                )
                return
            time.sleep(0.2)

    def shutdown(self) -> None:
        self._shutdown.set()

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Blocking control loop (executor must already spin in another thread)."""
        self.wait_for_clock()

        rate = self.create_rate(self.execution_freq)
        log_interval = 1
        if self.upsample and self.update_freq > 0:
            log_interval = max(int(round(self.execution_freq / self.update_freq)), 1)

        tick = 0
        perf0 = time.perf_counter()
        chunk_gaps_s: list = []
        last_chunk_end_t_s: Optional[float] = None
        waiting_logged = False

        try:
            while rclpy.ok() and not self._shutdown.is_set():
                will_infer = len(self.policy.action_queue) == 0
                obs = self.env.get_observations()
                if obs is None:
                    if self.upsample and (not will_infer):
                        obs = self.env.get_partial_observations()
                    if obs is None:
                        if not waiting_logged:
                            self.get_logger().info(
                                f"Waiting for observations: {', '.join(self.env.missing_observations())}"
                            )
                            waiting_logged = True
                        tick += 1
                        rate.sleep()
                        continue
                waiting_logged = False

                action = self.policy.act(obs)
                action_t_s = time.perf_counter() - perf0  # immediately after act()
                action_to_send = self.smoother.update(action)
                is_executed = self.env.execute_actions(action_to_send)
                sent_t_s = time.perf_counter() - perf0

                if is_executed:
                    if will_infer:
                        self.recorder.add_chunk_start(stamp_s=sent_t_s)
                        if last_chunk_end_t_s is not None:
                            gap = float(sent_t_s - last_chunk_end_t_s)
                            if gap >= 0:
                                chunk_gaps_s.append(gap)
                    # Last action of the current chunk was just consumed.
                    if (not will_infer) and len(self.policy.action_queue) == 0:
                        last_chunk_end_t_s = sent_t_s

                    if "joint_state" in obs:
                        self.recorder.add(
                            stamp_s=sent_t_s, joint_state=obs["joint_state"], action=action_to_send
                        )
                        if self.upsample and will_infer:
                            original_chunk = self.policy.get_last_original_action_chunk()
                            if original_chunk is not None and original_chunk.ndim == 2:
                                self.recorder.add_original_action_chunk(
                                    base_stamp_s=action_t_s,
                                    action_chunk=original_chunk,
                                    action_hz=self.update_freq,
                                )

                if tick % log_interval == 0:
                    self.get_logger().info(
                        ("Action executed." if is_executed else "Action NOT executed (control mode is not active).")
                        + f" instruction='{obs.get('instruction', '')}'"
                    )
                    self.get_logger().info(f"Action: {np.array2string(action, precision=3, suppress_small=True)}")

                if not self.upsample:
                    self.env.reset_observation()
                elif will_infer:
                    self.env.reset_observation(reset_joint_state=False)
                tick += 1
                rate.sleep()
        except KeyboardInterrupt:
            self.get_logger().info("KeyboardInterrupt received.")
        finally:
            self._log_chunk_gap_stats(chunk_gaps_s)
            self.policy.log_inference_stats()
            self.recorder.save_and_plot()
            try:
                self.env.stop_base()
            except Exception:
                pass

    def _log_chunk_gap_stats(self, chunk_gaps_s: list) -> None:
        if not chunk_gaps_s:
            self.get_logger().info("Chunk gap stats: no chunk gaps recorded.")
            return
        arr = np.asarray(chunk_gaps_s, dtype=np.float64)
        self.get_logger().info(
            "Chunk gap (prev chunk last action -> next chunk first action): "
            f"n={arr.size} mean={arr.mean() * 1e3:.1f}ms var={arr.var() * 1e6:.3f}(ms^2)"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HsrOpenpiNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
