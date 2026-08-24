"""Action-chunk handling around an openpi policy (ROS 2 port).

Mirrors the ``OpenpiPolicy`` class of the ROS 1 node: one inference produces an
action chunk, the first element is executed immediately and the rest is queued
(optionally upsampled to a higher execution rate). The model predicts *relative*
joint targets for the arm and head, so the current joint state is added back
before the command is sent.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from hsr_openpi.action_utils import (
    UPSAMPLE_METHOD_LINEAR,
    UPSAMPLE_METHOD_SPLINE,
    UPSAMPLE_METHODS,
    cubic_spline_upsample_actions,
    linear_upsample_actions,
)
from hsr_openpi.policy_client import BasePolicy


def _noop(msg: str) -> None:
    pass


class OpenpiPolicy:
    """Runs an openpi policy and turns its action chunks into per-tick commands."""

    def __init__(
        self,
        policy: BasePolicy,
        *,
        adopted_action_chunks: int = 15,
        action_hz: int = 10,
        upsample: bool = False,
        upsample_hz: int = 50,
        upsample_method: str = UPSAMPLE_METHOD_SPLINE,
        logwarn: Callable[[str], None] = _noop,
        loginfo: Callable[[str], None] = _noop,
    ):
        self.policy = policy
        self.adopted_action_chunks = int(adopted_action_chunks)
        self.action_hz = int(action_hz)
        self.upsample = bool(upsample)
        self.upsample_hz = int(upsample_hz)
        self.upsample_method = str(upsample_method)
        self._logwarn = logwarn
        self._loginfo = loginfo

        if self.upsample_method not in UPSAMPLE_METHODS:
            self._logwarn(
                f"Unknown upsample_method '{self.upsample_method}'. Falling back to "
                f"'{UPSAMPLE_METHOD_SPLINE}'. Available: {', '.join(UPSAMPLE_METHODS)}"
            )
            self.upsample_method = UPSAMPLE_METHOD_SPLINE

        self.execution_action_chunks = self.adopted_action_chunks
        if self.upsample:
            self.execution_action_chunks = max(
                int(round(self.adopted_action_chunks * float(self.upsample_hz) / float(self.action_hz))), 1
            )

        self.action_queue: deque = deque(maxlen=self.execution_action_chunks)
        self._last_original_action_chunk: Optional[np.ndarray] = None
        self._infer_latencies_s: List[float] = []

    # -- inference bookkeeping ------------------------------------------- #
    def _record_infer_timing(self, *, start_s: float, end_s: float) -> None:
        latency = float(end_s - start_s)
        if latency >= 0:
            self._infer_latencies_s.append(latency)

    def log_inference_stats(self) -> None:
        if not self._infer_latencies_s:
            self._loginfo("Inference stats: no inference calls recorded.")
            return
        arr = np.asarray(self._infer_latencies_s, dtype=np.float64)
        self._loginfo(
            f"Inference latency (infer() only): n={arr.size} "
            f"mean={arr.mean() * 1e3:.1f}ms var={arr.var() * 1e6:.3f}(ms^2)"
        )

    def get_last_original_action_chunk(self) -> Optional[np.ndarray]:
        return self._last_original_action_chunk

    def reset(self) -> None:
        self.action_queue.clear()
        self._last_original_action_chunk = None
        try:
            self.policy.reset()
        except Exception:  # pragma: no cover - optional on the backend
            pass

    # -- main entry point -------------------------------------------------- #
    @staticmethod
    def _delta_to_absolute(action: np.ndarray, joint_state: np.ndarray) -> np.ndarray:
        """Add the current joint state back to the relative arm/head targets.

        gripper (index 5) and base (indices 8..10) are absolute, so nothing is
        added for those dimensions.
        """
        offset = np.concatenate(
            [joint_state[:5], np.array([0.0], dtype=np.float32), joint_state[6:8], np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        )
        return action + offset

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return one 11-dim action.

        Parameters
        ----------
        obs : dict
            ``head_rgb`` (H, W, 3) uint8, ``hand_rgb`` (H, W, 3) uint8,
            ``joint_state`` (8,) float32, ``instruction`` str.

        Returns
        -------
        np.ndarray, shape (11,)
            ``[arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, gripper,
            head_pan, head_tilt, base_x, base_y, base_t]``
        """
        joint_state = np.asarray(obs["joint_state"], dtype=np.float32)

        if len(self.action_queue) > 0:
            return self._delta_to_absolute(self.action_queue.popleft(), joint_state)

        policy_input = {
            "head_rgb": obs["head_rgb"],
            "hand_rgb": obs["hand_rgb"],
            "state": joint_state,
            "prompt": obs["instruction"],
        }
        infer_start_s = time.perf_counter()
        raw_action_chunk = np.asarray(self.policy.infer(policy_input)["actions"], dtype=np.float32)
        self._record_infer_timing(start_s=infer_start_s, end_s=time.perf_counter())
        self._last_original_action_chunk = raw_action_chunk[: self.adopted_action_chunks]

        action_chunk = raw_action_chunk[: self.adopted_action_chunks]
        if self.upsample:
            upsampler = (
                linear_upsample_actions if self.upsample_method == UPSAMPLE_METHOD_LINEAR
                else cubic_spline_upsample_actions
            )
            action_chunk = upsampler(
                action_chunk,
                in_hz=self.action_hz,
                out_hz=self.upsample_hz,
                out_steps=self.execution_action_chunks,
            )

        self.action_queue.extend(action_chunk[1:])
        return self._delta_to_absolute(action_chunk[0], joint_state)
