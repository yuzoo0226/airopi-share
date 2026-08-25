#!/usr/bin/env python
"""Fast aggregation script — drop-in replacement for aggregate_stats_simple.py.

Optimizations over the simple version:
    1. Parquet I/O via PyArrow (column projection, no iterrows)
    2. Parallel directory loading with ThreadPoolExecutor
    3. Vectorized mixture-quantile computation across all dimensions simultaneously

Usage is identical to aggregate_stats_simple.py — same CLI arguments, same JSON output.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    from scripts.aggregate_stats_simple import (  # noqa: F401
        EpisodeStatsRecord,
        _approx_transform_mean_std,
        _assert_type_and_shape,
        _normalize,
        _required_feature_keys,
        _unnormalize,
        aggregate_feature_stats,
        aggregate_stats,
        combine_action_stats,
        combine_action_stats_arm_head_relative_gripper_base,
        compute_gripper_stats_with_action_transform,
        compute_gripper_stats_with_transform,
        load_episodes_stats,
        numpy_dict_to_lists,
        pad_hsr_action_stats,
        pad_hsr_state_stats,
    )
except ModuleNotFoundError as exc:
    # Allow both module execution (`python -m scripts.aggregate_stats_fast`)
    # and direct script execution (`python scripts/aggregate_stats_fast.py`).
    if exc.name != "scripts":
        raise
    from aggregate_stats_simple import (  # noqa: F401
        EpisodeStatsRecord,
        _approx_transform_mean_std,
        _assert_type_and_shape,
        _normalize,
        _required_feature_keys,
        _unnormalize,
        aggregate_feature_stats,
        aggregate_stats,
        combine_action_stats,
        combine_action_stats_arm_head_relative_gripper_base,
        compute_gripper_stats_with_action_transform,
        compute_gripper_stats_with_transform,
        load_episodes_stats,
        numpy_dict_to_lists,
        pad_hsr_action_stats,
        pad_hsr_state_stats,
    )

# ---------------------------------------------------------------------------
# Optimized PyArrow column-projection parquet reader
# ---------------------------------------------------------------------------

def load_gripper_data_from_parquet_dir(
    parquet_dir: str,
    state_column: str = "observation.state",
    action_column: str = "action.relative",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load per-episode gripper data (index 5) from a directory of parquet files.

    Uses PyArrow with column projection instead of pandas iterrows.
    """
    parquet_dir_path = Path(parquet_dir)
    if not parquet_dir_path.exists():
        raise ValueError(f"ディレクトリが存在しません: {parquet_dir}")

    parquet_files = sorted(glob.glob(str(parquet_dir_path / "episode_*.parquet")))
    if not parquet_files:
        raise ValueError(f"parquetファイルが見つかりません: {parquet_dir}")

    print(f"Found {len(parquet_files)} parquet files in {parquet_dir}")

    state_gripper_data_per_episode: list[np.ndarray] = []
    action_gripper_data_per_episode: list[np.ndarray] = []

    for parquet_file in parquet_files:
        try:
            # Read only needed columns via PyArrow
            schema = pq.read_schema(parquet_file)
            available_columns = set(schema.names)

            # State gripper
            if state_column in available_columns:
                table = pq.read_table(parquet_file, columns=[state_column])
                col_data = table[state_column].to_pylist()
                if col_data:
                    arr = np.array(col_data)
                    if arr.ndim == 2 and arr.shape[1] > 5:
                        state_gripper_data_per_episode.append(arr[:, 5].astype(float))
                    elif arr.ndim == 1:
                        # Each element is a list/array — stack then extract
                        vals = np.array([row[5] for row in col_data if len(row) > 5], dtype=float)
                        if vals.size > 0:
                            state_gripper_data_per_episode.append(vals)
            else:
                print(f"Warning: '{state_column}'column not found in {parquet_file}")

            # Action gripper
            if action_column in available_columns:
                table = pq.read_table(parquet_file, columns=[action_column])
                col_data = table[action_column].to_pylist()
                if col_data:
                    arr = np.array(col_data)
                    if arr.ndim == 2 and arr.shape[1] > 5:
                        action_gripper_data_per_episode.append(arr[:, 5].astype(float))
                    elif arr.ndim == 1:
                        vals = np.array([row[5] for row in col_data if len(row) > 5], dtype=float)
                        if vals.size > 0:
                            action_gripper_data_per_episode.append(vals)
            else:
                print(f"Warning: '{action_column}'column not found in {parquet_file}")

        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue

    print(f"Successfully loaded state gripper data from {len(state_gripper_data_per_episode)} episodes in {parquet_dir}")
    print(f"Successfully loaded action gripper data from {len(action_gripper_data_per_episode)} episodes in {parquet_dir}")
    return state_gripper_data_per_episode, action_gripper_data_per_episode


# ---------------------------------------------------------------------------
# Optimized : parallel directory loading
# ---------------------------------------------------------------------------

def load_gripper_data_from_multiple_dirs(
    parquet_dirs: list[str],
    state_column: str = "observation.state",
    action_column: str = "action.relative",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load gripper data from multiple directories in parallel using threads."""
    all_state_gripper_data: list[np.ndarray] = []
    all_action_gripper_data: list[np.ndarray] = []

    def _load_one(d: str):
        print(f"Processing directory: {d}")
        return load_gripper_data_from_parquet_dir(d, state_column, action_column)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(_load_one, parquet_dirs))

    for state_data, action_data in results:
        all_state_gripper_data.extend(state_data)
        all_action_gripper_data.extend(action_data)

    print(f"Total state gripper episodes loaded from {len(parquet_dirs)} directories: {len(all_state_gripper_data)}")
    print(f"Total action gripper episodes loaded from {len(parquet_dirs)} directories: {len(all_action_gripper_data)}")
    return all_state_gripper_data, all_action_gripper_data


# ---------------------------------------------------------------------------
# Optimized : fully-vectorized mixture quantile computation
# ---------------------------------------------------------------------------

def _std_normal_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal CDF using math.erf (no scipy needed)."""
    from math import erf as _m_erf
    return 0.5 * (1.0 + np.vectorize(_m_erf)(np.asarray(x, dtype=float) / sqrt(2.0)))


def _mixture_quantiles(
    per_episode: list[EpisodeStatsRecord],
    feature_key: str,
    q_levels: tuple[float, ...] = (0.01, 0.99),
) -> dict[float, list[float]]:
    """Compute quantiles of a Gaussian-mixture CDF, vectorized across dimensions.

    Each episode contributes a (truncated) normal component per dimension.
    The quantiles are found by bisection on the mixture CDF.
    """
    comps = []
    for rec in per_episode:
        ep_stats = rec.stats
        if feature_key not in ep_stats:
            continue
        s = ep_stats[feature_key]
        if not all(k in s for k in ("mean", "std", "min", "max", "count")):
            continue
        comps.append(s)

    if not comps:
        return {q: [] for q in q_levels}

    # shape (E, D)
    mean_arr = np.stack([c["mean"] for c in comps])
    std_arr = np.stack([c["std"] for c in comps])
    min_arr = np.stack([c["min"] for c in comps])
    max_arr = np.stack([c["max"] for c in comps])
    count_arr = np.stack([c["count"] for c in comps])  # (E, 1) or (E,)
    if count_arr.ndim > 1:
        count_arr = count_arr.reshape(count_arr.shape[0], -1)[:, 0]

    total_counts = count_arr.sum()
    weights = count_arr / (total_counts if total_counts > 0 else 1)  # (E,)

    E, dim = mean_arr.shape

    global_min = np.min(min_arr, axis=0)  # (D,)
    global_max = np.max(max_arr, axis=0)  # (D,)

    # Identify constant / non-finite dimensions
    constant_mask = global_min >= global_max  # (D,)
    nonfinite_mask = ~(np.isfinite(global_min) & np.isfinite(global_max))  # (D,)
    skip_mask = constant_mask | nonfinite_mask  # (D,)
    active_dims = np.where(~skip_mask)[0]

    results = {q: np.zeros(dim) for q in q_levels}
    # Fill constant dims
    for q in q_levels:
        results[q][constant_mask & ~nonfinite_mask] = global_min[constant_mask & ~nonfinite_mask]
        results[q][nonfinite_mask] = float("nan")

    if active_dims.size == 0:
        return {q: results[q].tolist() for q in q_levels}

    # Restrict arrays to active dims only: shapes (E, A) where A = len(active_dims)
    mu = mean_arr[:, active_dims]
    sig = std_arr[:, active_dims]
    lo = min_arr[:, active_dims]
    hi = max_arr[:, active_dims]

    # Point-mass vs continuous component masks: (E, A)
    point_mask = (sig <= 1e-12) | (hi <= lo)
    cont_mask = ~point_mask

    # Precompute truncated-normal denominators for continuous components
    # We'll compute Phi(a) and Phi(b) for all (E, A), but only use where cont_mask is True.
    # Use safe_sig to avoid division by zero (masked entries are irrelevant).
    safe_sig = np.where(cont_mask, sig, 1.0)
    a = (lo - mu) / safe_sig  # (E, A)
    b = (hi - mu) / safe_sig  # (E, A)
    Phi_a = _std_normal_cdf(a)  # (E, A)
    Phi_b = _std_normal_cdf(b)  # (E, A)
    denom = np.clip(Phi_b - Phi_a, 1e-12, None)  # (E, A)

    # weights: (E,) -> (E, 1) for broadcasting across dims
    w = weights[:, np.newaxis]  # (E, 1)
    # Zero out weights for point/cont separately
    w_point = np.where(point_mask, w, 0.0)  # (E, A)
    w_cont = np.where(cont_mask, w, 0.0)    # (E, A)

    def cdf_vec(x: np.ndarray) -> np.ndarray:
        """Evaluate mixture CDF at x for all active dims. x shape: (A,), returns (A,)."""
        # Point-mass contribution: sum_e w_e * (x >= mu_e)
        x_2d = x[np.newaxis, :]  # (1, A)
        total = np.sum(w_point * (x_2d >= mu), axis=0)  # (A,)
        # Continuous contribution: truncated normal
        z = (x_2d - mu) / safe_sig  # (E, A)
        Phi_z = _std_normal_cdf(z)  # (E, A)
        p = np.clip((Phi_z - Phi_a) / denom, 0.0, 1.0)  # (E, A)
        p = np.where(x_2d <= lo, 0.0, p)
        p = np.where(x_2d >= hi, 1.0, p)
        total = total + np.sum(w_cont * p, axis=0)  # (A,)
        return total

    # Bisection: operate on all active dims simultaneously
    for q in q_levels:
        left = global_min[active_dims].copy()
        right = global_max[active_dims].copy()
        for _ in range(50):
            mid = 0.5 * (left + right)
            cdf_val = cdf_vec(mid)
            go_right = cdf_val < q
            left = np.where(go_right, mid, left)
            right = np.where(go_right, right, mid)
        results[q][active_dims] = 0.5 * (left + right)

    return {q: results[q].tolist() for q in q_levels}


# ---------------------------------------------------------------------------
# Quantile-combining helpers
# (defined inside main() in aggregate_stats_simple.py, so kept here)
# ---------------------------------------------------------------------------

def _combine_action_quantiles(
    state_qs: dict[float, list[float]] | None,
    relative_qs: dict[float, list[float]] | None,
    base_dim: int = 3,
) -> dict[float, list[float]]:
    if not state_qs or not state_qs.get(0.01):
        return relative_qs or {0.01: [], 0.99: []}
    if not relative_qs or not relative_qs.get(0.01):
        return state_qs
    state_dim = len(state_qs[0.01])
    relative_dim = len(relative_qs[0.01])
    if state_dim == relative_dim:
        return state_qs
    if state_dim + base_dim != relative_dim:
        raise ValueError(
            f"Cannot combine action quantiles: state_diff dim={state_dim}, relative dim={relative_dim}, base_dim={base_dim}"
        )
    combined = {}
    for q in (0.01, 0.99):
        combined[q] = list(state_qs.get(q, [])) + list(relative_qs.get(q, [])[-base_dim:])
    return combined


def _combine_action_quantiles_arm_head_relative_gripper_base(
    state_qs: dict[float, list[float]] | None,
    relative_qs: dict[float, list[float]] | None,
    *,
    arm_dim: int = 5,
    gripper_index: int = 5,
    head_dim: int = 2,
    base_dim: int = 3,
) -> dict[float, list[float]]:
    if not state_qs or not state_qs.get(0.01):
        return relative_qs or {0.01: [], 0.99: []}
    if not relative_qs or not relative_qs.get(0.01):
        return state_qs
    state_dim = len(state_qs[0.01])
    relative_dim = len(relative_qs[0.01])
    expected_state_dim = arm_dim + 1 + head_dim
    expected_relative_dim = expected_state_dim + base_dim
    if state_dim != expected_state_dim or relative_dim != expected_relative_dim:
        raise ValueError(
            "Cannot combine action quantiles: "
            f"state_diff dim={state_dim}, relative dim={relative_dim}, "
            f"expected_state_dim={expected_state_dim}, expected_relative_dim={expected_relative_dim}"
        )
    combined = {}
    for q in (0.01, 0.99):
        left = list(state_qs.get(q, []))[:arm_dim]
        gripper = list(relative_qs.get(q, []))[gripper_index : gripper_index + 1]
        head_start = gripper_index + 1
        head = list(state_qs.get(q, []))[head_start : head_start + head_dim]
        base = list(relative_qs.get(q, []))[-base_dim:]
        combined[q] = left + gripper + head + base
    return combined


# ---------------------------------------------------------------------------
# main() — identical logic to aggregate_stats_simple.py
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-stats", required=True, help="Path to episodes_stats.jsonl")
    ap.add_argument("--output-file", required=True, help="Path to output aggregated JSON")
    ap.add_argument("--no-adapt-to-pi", action="store_true", help="HSR gripper角度空間への変換を無効化 (state のみ)")
    ap.add_argument(
        "--convert-gripper",
        action="store_true",
        help="Apply gripper conversion functions when computing stats (default: off)",
    )
    ap.add_argument("--chunk-dir", required=True, help="Path(s) to data directory(ies) containing chunk* directories")
    ap.add_argument("--state-column", default="observation.state", help="Column name containing state gripper data")
    ap.add_argument("--action-column", default="action.relative", help="Column name containing action gripper data")
    ap.add_argument(
        "--action-mode",
        default="relative",
        choices=(
            "relative",
            "absolute_arm_head_relative_gripper_base",
            "state_diff_arm_head_relative_gripper_base",
        ),
        help="Action source to use for stats (default: relative)",
    )
    ap.add_argument(
        "--include-all-features",
        action="store_true",
        help="If set, aggregate all features from episodes_stats.jsonl (default: only required state/action keys).",
    )
    ap.add_argument("--write-meta", action="store_true", help="出力JSONに meta(transform設定) を含める")
    ap.add_argument(
        "--min-std",
        type=float,
        default=0.01,
        help="Floor for the per-dimension standard deviation. Normalisation divides by "
        "(std + 1e-6), so a joint the task holds constant -- arm_roll and wrist_roll are "
        "commanded to a fixed 0.0 here -- turns the controller's own jitter into enormous "
        "normalised values: measured std 0.0002 gives |z| up to 235, and 235^2 lands in the "
        "loss as a 5e4 spike. A dataset where those joints never move at all is safe, because "
        "the numerator is exactly zero too; add enough episodes for a little jitter and it is "
        "not. Set to 0 to disable.",
    )
    args = ap.parse_args()

    include_features = None
    if not args.include_all_features:
        include_features = _required_feature_keys(args.action_mode, args.action_column)
    stats_records = load_episodes_stats(args.episodes_stats, include_features=include_features)

    if args.convert_gripper:
        print(f"Loading gripper data from chunk directories: {args.chunk_dir}")
        chunk_dirs = []
        data_path = Path(args.chunk_dir)
        if data_path.exists():
            found_chunks = sorted(glob.glob(str(data_path / "chunk-*")))
            if found_chunks:
                chunk_dirs.extend(found_chunks)
                print(
                    f"Found {len(found_chunks)} chunk directories in {args.chunk_dir}: "
                    f"{[Path(p).name for p in found_chunks]}"
                )
            else:
                print(f"No chunk-* directories found in {args.chunk_dir}")
        else:
            print(f"Warning: Data directory does not exist: {args.chunk_dir}")

        if chunk_dirs:
            state_gripper_data, action_gripper_data = load_gripper_data_from_multiple_dirs(
                chunk_dirs, args.state_column, args.action_column
            )
            state_gripper_stats = None
            if state_gripper_data:
                state_gripper_stats = compute_gripper_stats_with_transform(
                    state_gripper_data,
                    convert_gripper=args.convert_gripper,
                )
                print(
                    f"State gripper stats: mean={state_gripper_stats['mean'][0]:.6f}, "
                    f"std={state_gripper_stats['std'][0]:.6f}"
                )
            action_gripper_stats = None
            if action_gripper_data:
                action_gripper_stats = compute_gripper_stats_with_action_transform(
                    action_gripper_data,
                    convert_gripper=args.convert_gripper,
                )
                print(
                    "Action gripper stats (with _gripper_from_angular_inv transform): "
                    f"mean={action_gripper_stats['mean'][0]:.6f}, std={action_gripper_stats['std'][0]:.6f}"
                )
        else:
            print("No chunk directories found, skipping gripper processing")
            state_gripper_stats = None
            action_gripper_stats = None
    else:
        print("Gripper conversion disabled; skipping chunk directory scan.")
        state_gripper_stats = None
        action_gripper_stats = None

    if state_gripper_stats or action_gripper_stats:
        for i, rec in enumerate(stats_records):
            ep_stats = rec.stats
            if state_gripper_stats and "observation.state" in ep_stats:
                state_stats = ep_stats["observation.state"]
                for key in ["min", "max", "mean", "std"]:
                    if key in state_stats and len(state_stats[key]) > 5:
                        state_stats[key][5] = state_gripper_stats[key][0]
            if action_gripper_stats and args.action_column in ep_stats:
                action_stats = ep_stats[args.action_column]
                for key in ["min", "max", "mean", "std"]:
                    if key in action_stats and len(action_stats[key]) > 5:
                        action_stats[key][5] = action_gripper_stats[key][0]
    else:
        print("Gripper stats not updated (no chunk directories processed)")

    agg = aggregate_stats(stats_records)
    agg_lists = numpy_dict_to_lists(agg)

    adapt = not args.no_adapt_to_pi
    adapt_state = adapt
    if "observation.state" not in agg:
        raise KeyError("episodes_statsに'observation.state'が必要です")
    if "action.relative" not in agg:
        raise KeyError("episodes_statsに'action.relative'が必要です")
    if args.action_mode == "absolute_arm_head_relative_gripper_base" and "action.absolute" not in agg:
        raise KeyError("episodes_statsに'action.absolute'が必要です")
    if args.action_mode == "state_diff_arm_head_relative_gripper_base" and "action.state_diff" not in agg:
        raise KeyError("episodes_statsに'action.state_diff'が必要です")

    padded_state = pad_hsr_state_stats(agg["observation.state"], adapt_to_pi=adapt_state)

    if args.action_mode == "relative":
        combined_action_stats = agg["action.relative"]
    elif args.action_mode == "absolute_arm_head_relative_gripper_base":
        combined_action_stats = combine_action_stats_arm_head_relative_gripper_base(
            agg.get("action.absolute"),
            agg["action.relative"],
        )
    elif args.action_mode == "state_diff_arm_head_relative_gripper_base":
        combined_action_stats = combine_action_stats_arm_head_relative_gripper_base(
            agg.get("action.state_diff"),
            agg["action.relative"],
        )
    padded_actions = pad_hsr_action_stats(combined_action_stats, adapt_to_pi=adapt)

    # Quantiles
    state_qs = _mixture_quantiles(stats_records, "observation.state")
    if args.action_mode == "relative":
        action_qs = _mixture_quantiles(stats_records, "action.relative")
    elif args.action_mode == "absolute_arm_head_relative_gripper_base":
        action_qs = _combine_action_quantiles_arm_head_relative_gripper_base(
            _mixture_quantiles(stats_records, "action.absolute"),
            _mixture_quantiles(stats_records, "action.relative"),
        )
    elif args.action_mode == "state_diff_arm_head_relative_gripper_base":
        action_qs = _combine_action_quantiles_arm_head_relative_gripper_base(
            _mixture_quantiles(stats_records, "action.state_diff"),
            _mixture_quantiles(stats_records, "action.relative"),
        )

    state_aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12]
    action_aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]
    if state_qs[0.01]:
        padded_state["q01"] = [0.0] * len(padded_state.get("mean", []))
        padded_state["q99"] = [0.0] * len(padded_state.get("mean", []))
        for src_i, dst_i in enumerate(state_aligned_ids):
            if dst_i < len(padded_state["q01"]):
                padded_state["q01"][dst_i] = float(state_qs[0.01][src_i])
                padded_state["q99"][dst_i] = float(state_qs[0.99][src_i])
    if action_qs[0.01]:
        padded_actions["q01"] = [0.0] * len(padded_actions.get("mean", []))
        padded_actions["q99"] = [0.0] * len(padded_actions.get("mean", []))
        for src_i, dst_i in enumerate(action_aligned_ids):
            if dst_i < len(padded_actions["q01"]):
                padded_actions["q01"][dst_i] = float(action_qs[0.01][src_i])
                padded_actions["q99"][dst_i] = float(action_qs[0.99][src_i])

    agg_lists["state"] = padded_state
    agg_lists["actions"] = padded_actions

    def _finalize(d: dict) -> dict:
        return {k: d.get(k) for k in ("mean", "std", "q01", "q99")}

    agg_lists["state"] = _finalize(agg_lists["state"])
    agg_lists["actions"] = _finalize(agg_lists["actions"])

    out = {"norm_stats": {"state": agg_lists["state"], "actions": agg_lists["actions"]}}

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Type {type(o)} not serializable")

    if args.min_std > 0:
        for group in out.get("norm_stats", {}).values():
            std = np.asarray(group["std"], dtype=np.float64)
            # Leave exact zeros alone: those dimensions are padding or truly
            # constant, and openpi already handles them (numerator is zero too).
            raised = (std > 0) & (std < args.min_std)
            if raised.any():
                print(
                    f"[INFO] raising {int(raised.sum())} std value(s) below {args.min_std} "
                    f"(min was {std[std > 0].min():.2e})"
                )
                std[raised] = args.min_std
                group["std"] = std.tolist()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_file).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default))
    print(f"[OK] wrote {args.output_file}")


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
