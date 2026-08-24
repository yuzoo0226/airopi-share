"""Action post-processing helpers shared by the ROS 2 openpi HSR deployment.

Ported from the ROS 1 node (``deploy/hsr_openpi_deploy/scripts/hsr_openpi.py``).
The numerics are unchanged; the only difference is that logging goes through an
injected callable instead of ``rospy``.
"""

from collections import deque
from typing import Callable, Optional

import numpy as np

UPSAMPLE_METHOD_SPLINE = "spline"
UPSAMPLE_METHOD_LINEAR = "linear"
UPSAMPLE_METHODS = [UPSAMPLE_METHOD_SPLINE, UPSAMPLE_METHOD_LINEAR]

ACTION_SMOOTHING_NONE = "none"
ACTION_SMOOTHING_EMA = "ema"
ACTION_SMOOTHING_MA = "moving_average"
ACTION_SMOOTHING_METHODS = [ACTION_SMOOTHING_NONE, ACTION_SMOOTHING_EMA, ACTION_SMOOTHING_MA]

MODE_CONTINUOUS = "continuous"
MODE_DISCRETE = "discrete"
MODE_HYBRID = "hybrid"
GRIPPER_MODES = [MODE_CONTINUOUS, MODE_DISCRETE, MODE_HYBRID]


def _default_logger(msg: str) -> None:
    print(msg)


def natural_cubic_spline_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
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
        rhs = 3.0 * ((y[2:] - y[1:-1]) / h[1:, None] - (y[1:-1] - y[:-2]) / h[:-1, None])  # (m, dim)

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


def linear_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Piecewise linear interpolation."""
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


def _upsample(
    actions: np.ndarray,
    *,
    in_hz: float,
    out_hz: float,
    out_steps: Optional[int],
    interpolator: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
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
    return interpolator(x, y, xq).astype(np.float32, copy=False)


def cubic_spline_upsample_actions(
    actions: np.ndarray, *, in_hz: float, out_hz: float, out_steps: Optional[int] = None
) -> np.ndarray:
    """Upsample an action sequence with natural cubic spline interpolation."""
    return _upsample(
        actions, in_hz=in_hz, out_hz=out_hz, out_steps=out_steps, interpolator=natural_cubic_spline_interpolate
    )


def linear_upsample_actions(
    actions: np.ndarray, *, in_hz: float, out_hz: float, out_steps: Optional[int] = None
) -> np.ndarray:
    """Upsample an action sequence with linear interpolation."""
    return _upsample(actions, in_hz=in_hz, out_hz=out_hz, out_steps=out_steps, interpolator=linear_interpolate)


class ActionSmoother:
    """EMA / moving-average smoother applied to a subset of the action dims."""

    def __init__(
        self,
        *,
        method: str,
        ema_alpha: float,
        ma_window: int,
        dims_mask: np.ndarray,
        logwarn: Callable[[str], None] = _default_logger,
    ):
        self.method = str(method)
        if self.method not in ACTION_SMOOTHING_METHODS:
            logwarn(
                f"Unknown action_smoothing '{self.method}'. Falling back to '{ACTION_SMOOTHING_NONE}'. "
                f"Available: {', '.join(ACTION_SMOOTHING_METHODS)}"
            )
            self.method = ACTION_SMOOTHING_NONE

        self.ema_alpha = float(np.clip(float(ema_alpha), 0.0, 1.0))
        self.ma_window = max(int(ma_window), 1)
        self.dims_mask = np.asarray(dims_mask, dtype=bool).reshape(-1)

        self._ema_state: Optional[np.ndarray] = None
        self._ma_buf: deque = deque(maxlen=self.ma_window)

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

    def reset(self) -> None:
        self._ema_state = None
        self._ma_buf.clear()


def build_trace_group_name(
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

    parts: list = [f"ac{int(adopted_action_chunks)}", f"uf{int(update_freq)}"]

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
