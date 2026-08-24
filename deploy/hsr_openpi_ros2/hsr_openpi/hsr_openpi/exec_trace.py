"""Execution trace recorder (ROS 2 port of ``ExecTraceRecorder``).

Stores the executed action / measured joint state time series of one rollout as
``<base_dir>/<config>/NNNN.npz`` plus a quick-look plot ``NNNN.png``.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional

import numpy as np


def _print(msg: str) -> None:
    print(msg)


class ExecTraceRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        config_name: str,
        joint_dim_names: Optional[List[str]] = None,
        base_action_names: Optional[List[str]] = None,
        base_dir: str = "/home/hsr/deploy_record",
        loginfo: Callable[[str], None] = _print,
        logwarn: Callable[[str], None] = _print,
    ):
        self.enabled = bool(enabled)
        self.config_name = str(config_name)
        self.joint_dim_names = list(joint_dim_names or [])
        self.base_action_names = list(base_action_names or [])
        self.base_dir = str(base_dir)
        self._loginfo = loginfo
        self._logwarn = logwarn

        self._t: List[float] = []
        self._joint_state: List[np.ndarray] = []
        self._action: List[np.ndarray] = []
        self._t_action_original: List[float] = []
        self._action_original_delta: List[np.ndarray] = []
        self._t_chunk_start: List[float] = []
        self._saved = False

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

    @staticmethod
    def _next_run_index(out_dir: str) -> int:
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
        if not self.enabled or self._saved:
            return
        if len(self._t) == 0:
            self._logwarn("ExecTraceRecorder: no samples to save.")
            return
        self._saved = True

        out_dir = self._output_dir()
        os.makedirs(out_dir, exist_ok=True)
        run_idx = self._next_run_index(out_dir)
        stem = f"{run_idx:04d}"

        npz_path = os.path.join(out_dir, f"{stem}.npz")
        plot_path = os.path.join(out_dir, f"{stem}.png")

        payload = {
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
        self._loginfo(f"Saved exec trace: {npz_path}")

        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except Exception as e:
            self._logwarn(f"ExecTraceRecorder: matplotlib unavailable, skipping plot ({e}).")
            return

        t = np.asarray(self._t, dtype=np.float64)
        joint_state = np.stack(self._joint_state, axis=0)
        action = np.stack(self._action, axis=0)
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
                if i < 5 or 6 <= i < 8:
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
        self._loginfo(f"Saved exec trace plot: {plot_path}")
