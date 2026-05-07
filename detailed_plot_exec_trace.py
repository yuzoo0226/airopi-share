#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_JOINT_STATE_NAMES = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "hand_motor_joint",
    "head_pan_joint",
    "head_tilt_joint",
]
DEFAULT_BASE_ACTION_NAMES = ["base_x", "base_y", "base_theta"]


@dataclass(frozen=True)
class TimeRange:
    start_s: float
    end_s: float

    def normalized(self) -> "TimeRange":
        start_s = float(self.start_s)
        end_s = float(self.end_s)
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        return TimeRange(start_s=start_s, end_s=end_s)


def _parse_time_range(value: str) -> TimeRange:
    # Accept forms: "0,5" / "0:5" / "(0, 5)"
    s = value.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    elif ":" in s:
        parts = [p.strip() for p in s.split(":") if p.strip() != ""]
    else:
        raise argparse.ArgumentTypeError(f"Invalid range '{value}'. Use 'start,end' (e.g. 0.0,5.0).")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid range '{value}'. Use 'start,end' (e.g. 0.0,5.0).")
    try:
        start_s = float(parts[0])
        end_s = float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid range '{value}'. start/end must be floats.") from e
    return TimeRange(start_s=start_s, end_s=end_s).normalized()


def _safe_float_tag(x: float) -> str:
    # For filenames: -1.234 -> m1p234
    sign = "m" if x < 0 else ""
    x = abs(float(x))
    return f"{sign}{x:.3f}".replace(".", "p")


def _infer_stem_from_npz_path(npz_path: str) -> str:
    base = os.path.basename(npz_path)
    stem = base[:-4] if base.endswith(".npz") else os.path.splitext(base)[0]
    m = re.match(r"^(\d+)$", stem)
    return m.group(1) if m else stem


def _output_dir_for_npz(npz_path: str, *, suffix: str = "") -> str:
    parent = os.path.dirname(os.path.abspath(npz_path))
    stem = _infer_stem_from_npz_path(npz_path)
    return os.path.join(parent, f"detailed_plot_{stem}{suffix}")


def _ms_tag(ms: float) -> str:
    # For directory names. Examples: 80 -> "80", 80.5 -> "80p5", -20 -> "m20"
    s = f"{float(ms):g}"
    return s.replace("-", "m").replace(".", "p")


def _load_names(data: np.lib.npyio.NpzFile, *, action_dim: int, joint_dim: int) -> tuple[list[str], list[str]]:
    # Returns (joint_names, action_names). Uses embedded names if available, otherwise falls back.
    joint_names = None
    base_action_names = None

    if "joint_dim_names" in data:
        try:
            joint_names = [str(x) for x in data["joint_dim_names"].tolist()]
        except Exception:
            joint_names = None
    if "base_action_names" in data:
        try:
            base_action_names = [str(x) for x in data["base_action_names"].tolist()]
        except Exception:
            base_action_names = None

    joint_names = joint_names or list(DEFAULT_JOINT_STATE_NAMES)
    base_action_names = base_action_names or list(DEFAULT_BASE_ACTION_NAMES)

    # Trim/pad to match actual shapes (best-effort).
    joint_names = joint_names[:joint_dim] + [f"joint[{i}]" for i in range(len(joint_names), joint_dim)]

    action_names: list[str] = []
    for i in range(int(action_dim)):
        if i < len(joint_names):
            action_names.append(joint_names[i])
            continue
        base_i = i - len(joint_names)
        if 0 <= base_i < len(base_action_names):
            action_names.append(base_action_names[base_i])
            continue
        action_names.append(f"action[{i}]")
    return joint_names, action_names


def _apply_100ms_grid(ax) -> None:
    from matplotlib.ticker import MultipleLocator

    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="major", alpha=0.35, linewidth=0.6)
    ax.grid(which="minor", alpha=0.18, linewidth=0.4)


def _plot(
    *,
    t_s: np.ndarray,
    joint_state: np.ndarray,
    action: np.ndarray,
    t_action_original_s: np.ndarray | None,
    action_original: np.ndarray | None,
    t_chunk_start_s: np.ndarray | None,
    joint_names: list[str],
    action_names: list[str],
    out_path: str,
    xlim: tuple[float, float] | None,
    shift_joint: bool,
    joint_shift_s: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if t_s.ndim != 1:
        raise ValueError("t must be 1D.")
    if joint_state.ndim != 2 or action.ndim != 2:
        raise ValueError("joint_state/action must be 2D.")
    if (t_action_original_s is None) != (action_original is None):
        raise ValueError("t_action_original_s and action_original must be both provided or both None.")
    if t_action_original_s is not None and action_original is not None:
        if t_action_original_s.ndim != 1 or action_original.ndim != 2:
            raise ValueError("t_action_original_s must be 1D and action_original must be 2D.")
        if action_original.shape[0] != t_action_original_s.shape[0]:
            raise ValueError("t_action_original_s and action_original must have the same length.")
    if t_chunk_start_s is not None:
        t_chunk_start_s = np.asarray(t_chunk_start_s, dtype=np.float64).reshape(-1)

    n_joint = int(joint_state.shape[1])
    n_action = int(action.shape[1])
    n_action_original = int(action_original.shape[1]) if action_original is not None else 0
    nrows = max(n_joint, n_action, 1)

    fig_h = max(6.0, 1.15 * float(nrows))
    fig, axes = plt.subplots(nrows, 1, figsize=(16, fig_h), sharex=True)
    if nrows == 1:
        axes = [axes]

    # Create stable legend handles (only once).
    joint_style = dict(linestyle="None", marker=".", markersize=2.2, alpha=0.95)
    action_style = dict(linestyle="None", marker="x", markersize=2.8, alpha=0.95)
    original_style = dict(linestyle="None", marker="+", markersize=4.0, alpha=0.95)

    joint_shift_s = float(joint_shift_s)
    t_joint_s = t_s - joint_shift_s if shift_joint else t_s

    for i in range(nrows):
        ax = axes[i]
        has_joint = i < n_joint
        has_action = i < n_action

        if has_joint:
            ax.plot(t_joint_s, joint_state[:, i], label="joint", **joint_style)
        if has_action:
            ax.plot(t_s, action[:, i], label="action", **action_style)
        if t_action_original_s is not None and action_original is not None and i < n_action_original:
            ax.plot(t_action_original_s, action_original[:, i], label="original action", **original_style)
        dim_name = action_names[i] if has_action else joint_names[i] if has_joint else f"dim[{i}]"
        ax.set_ylabel(dim_name)
        _apply_100ms_grid(ax)

        if i == 0:
            ax.set_title("exec trace (joint: '.', action: 'x', original action: '+')")
        if has_joint or has_action:
            # Only show "joint/action" legend (no per-dim entries).
            handles, labels = ax.get_legend_handles_labels()
            dedup = {}
            for h, l in zip(handles, labels, strict=False):
                dedup.setdefault(l, h)
            ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)

    if xlim is not None:
        axes[-1].set_xlim(xlim)
    axes[-1].set_xlabel("time [s] (relative)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, format="png")
    plt.close(fig)


def _slice_by_time(
    t_s: np.ndarray, joint_state: np.ndarray, action: np.ndarray, *, start_s: float, end_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_s = float(start_s)
    end_s = float(end_s)
    mask = (t_s >= start_s) & (t_s <= end_s)
    return t_s[mask], joint_state[mask], action[mask]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create detailed plots from an exec-trace .npz (saved by hsr_openpi). "
            "Outputs to detailed_plot_{stem}/ next to the .npz."
        )
    )
    parser.add_argument("npz", help="Path to exec trace .npz (e.g. /home/openpi/deploy_record/<config>/0001.npz)")
    parser.add_argument(
        "--range",
        dest="ranges",
        action="append",
        default=[],
        type=_parse_time_range,
        help="Zoom range in seconds (relative): 'start,end' (repeatable), e.g. --range 0.0,5.0",
    )
    parser.add_argument(
        "--shift-joint",
        action="store_true",
        help="Shift joint timestamps earlier (left) by --joint-shift-ms for plotting only.",
    )
    parser.add_argument(
        "--joint-shift-ms",
        type=float,
        default=80.0,
        help="Joint timestamp shift amount in milliseconds (positive shifts earlier/left). Default: 80.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    npz_path = os.path.abspath(args.npz)
    out_dir_suffix = f"_shift_{_ms_tag(args.joint_shift_ms)}" if args.shift_joint else ""
    out_dir = _output_dir_for_npz(npz_path, suffix=out_dir_suffix)
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=False)
    if "t" not in data or "joint_state" not in data or "action" not in data:
        raise KeyError("npz must contain keys: 't', 'joint_state', 'action'.")

    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    joint_state = np.asarray(data["joint_state"], dtype=np.float32)
    action = np.asarray(data["action"], dtype=np.float32)
    if joint_state.ndim != 2 or action.ndim != 2:
        raise ValueError("joint_state/action must be 2D arrays.")
    if joint_state.shape[0] != t.shape[0] or action.shape[0] != t.shape[0]:
        raise ValueError("t, joint_state, and action must have the same length (T).")

    # `hsr_openpi.py` stores timestamps as perf_counter() deltas starting from 0.
    # Keep them as-is so that `t_action_original` can precede `t` slightly.
    t_rel = t

    joint_names, action_names = _load_names(data, action_dim=int(action.shape[1]), joint_dim=int(joint_state.shape[1]))

    t_action_original = None
    action_original = None
    if "t_action_original" in data and ("action_original_delta" in data or "action_original" in data):
        t_action_original = np.asarray(data["t_action_original"], dtype=np.float64).reshape(-1)
        if "action_original_delta" in data:
            action_original_delta = np.asarray(data["action_original_delta"], dtype=np.float32)
            if action_original_delta.ndim == 2 and action_original_delta.shape[0] == t_action_original.shape[0]:
                # Convert delta->command using joint_state sampled at the closest previous time.
                idx = np.searchsorted(t_rel, t_action_original, side="right") - 1
                idx = np.clip(idx, 0, max(len(t_rel) - 1, 0))
                js = joint_state[idx]
                action_original = np.array(action_original_delta, copy=True)
                # arm/head are deltas; gripper/base are already absolute/command values.
                if action_original.shape[1] >= 5 and js.shape[1] >= 5:
                    action_original[:, :5] = action_original[:, :5] + js[:, :5]
                if action_original.shape[1] >= 8 and js.shape[1] >= 8:
                    action_original[:, 6:8] = action_original[:, 6:8] + js[:, 6:8]
            else:
                t_action_original = None
        else:
            action_original = np.asarray(data["action_original"], dtype=np.float32)
            if action_original.ndim != 2 or action_original.shape[0] != t_action_original.shape[0]:
                t_action_original = None
                action_original = None
    t_action_original_rel = t_action_original if t_action_original is not None and action_original is not None else None

    t_chunk_start = None
    if "t_chunk_start" in data:
        t_chunk_start = np.asarray(data["t_chunk_start"], dtype=np.float64).reshape(-1)

    meta = {
        "input_npz": npz_path,
        "output_dir": out_dir,
        "t0_abs": float(t[0]),
        "t_end_abs": float(t[-1]),
        "duration_s": float(t_rel[-1]) if len(t_rel) else 0.0,
        "joint_dim": int(joint_state.shape[1]),
        "action_dim": int(action.shape[1]),
        "has_original_action": bool(t_action_original_rel is not None and action_original is not None),
        "original_action_points": int(len(t_action_original_rel)) if t_action_original_rel is not None else 0,
        "chunk_starts": int(len(t_chunk_start)) if t_chunk_start is not None else 0,
        "ranges": [{"start_s": r.start_s, "end_s": r.end_s} for r in args.ranges],
        "shift_joint": bool(args.shift_joint),
        "joint_shift_ms": float(args.joint_shift_ms),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    joint_shift_s = float(args.joint_shift_ms) / 1000.0

    # Full plot
    _plot(
        t_s=t_rel,
        joint_state=joint_state,
        action=action,
        t_action_original_s=t_action_original_rel,
        action_original=action_original,
        t_chunk_start_s=t_chunk_start,
        joint_names=joint_names,
        action_names=action_names,
        out_path=os.path.join(out_dir, "all.png"),
        xlim=None,
        shift_joint=bool(args.shift_joint),
        joint_shift_s=joint_shift_s,
    )

    # Zoom plots
    for r in args.ranges:
        t_z, joint_z, action_z = _slice_by_time(t_rel, joint_state, action, start_s=r.start_s, end_s=r.end_s)
        if t_action_original_rel is not None and action_original is not None:
            mask_o = (t_action_original_rel >= r.start_s) & (t_action_original_rel <= r.end_s)
            t_oz = t_action_original_rel[mask_o]
            action_oz = action_original[mask_o]
        else:
            t_oz = None
            action_oz = None
        if t_chunk_start is not None:
            t_cz = t_chunk_start[(t_chunk_start >= r.start_s) & (t_chunk_start <= r.end_s)]
        else:
            t_cz = None
        if len(t_z) == 0:
            continue
        tag = f"{_safe_float_tag(r.start_s)}_{_safe_float_tag(r.end_s)}"
        _plot(
            t_s=t_z,
            joint_state=joint_z,
            action=action_z,
            t_action_original_s=t_oz,
            action_original=action_oz,
            t_chunk_start_s=t_cz,
            joint_names=joint_names,
            action_names=action_names,
            out_path=os.path.join(out_dir, f"zoom_{tag}.png"),
            xlim=(r.start_s, r.end_s),
            shift_joint=bool(args.shift_joint),
            joint_shift_s=joint_shift_s,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
