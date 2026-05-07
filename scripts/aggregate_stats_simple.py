#!/usr/bin/env python
"""Simple aggregation script using only LeRobot aggregate_stats.

Reads an episodes_stats.jsonl (each line: {"episode_index": int, "stats": {feature_key: {min,max,mean,std,count}}})
Converts lists to numpy arrays, applies aggregate_stats, and writes result as JSON (lists again).

Enhanced to support gripper-specific processing:
- Gripper data can be loaded from parquet files and transformed with nonlinear functions
- Non-gripper data uses episodes_stats.jsonl as-is

Usage:
  # Basic usage (original behavior)
  uv run python scripts/aggregate_stats_simple.py \
    --episodes-stats datasets/lerobot_datasets/weblab_params_debug/meta/episodes_stats.jsonl \
    --output-file /tmp/aggregated_stats.json

  # With gripper data from chunk directories (auto-detects chunk-* subdirectories)
  # Processes both state and action gripper data from parquet files
  uv run python scripts/aggregate_stats_simple.py \
    --episodes-stats datasets/lerobot_datasets/weblab_params_debug/meta/episodes_stats.jsonl \
    --chunk-dir /path/to/lerobot_datasets/2025-06-v2.1_filtered/data \
    --state-column observation.state \
    --action-column action.relative \
    --output-file /tmp/aggregated_stats.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
# from lerobot.common.datasets.compute_stats import aggregate_stats

# Import gripper transformation functions from hsr_policy.py
from openpi.policies.hsr_policy import _gripper_to_angular, _gripper_from_angular_inv


@dataclasses.dataclass(frozen=True)
class EpisodeStatsRecord:
    """One line in episodes_stats.jsonl with source metadata for debugging."""

    line_no: int
    episode_index: int | None
    episode_file: str | None
    stats: dict[str, dict[str, np.ndarray]]


def _shape_of_array(v: np.ndarray) -> tuple[int, ...]:
    return tuple(np.asarray(v).shape)


def _format_record_context(rec: EpisodeStatsRecord) -> str:
    parts = [f"line={rec.line_no}"]
    if rec.episode_index is not None:
        parts.append(f"episode_index={rec.episode_index}")
    if rec.episode_file:
        parts.append(f"episode_file={rec.episode_file}")
    return ", ".join(parts)


def _stack_with_debug(
    values: list[np.ndarray],
    *,
    feature_key: str,
    metric: str,
    contexts: list[EpisodeStatsRecord] | None = None,
) -> np.ndarray:
    """np.stack with enriched diagnostics on shape mismatch."""

    try:
        return np.stack(values)
    except ValueError as exc:
        if "same shape" not in str(exc):
            raise
        shapes = [_shape_of_array(v) for v in values]
        shape_counts = Counter(shapes)
        shape_summary = ", ".join(f"{shape}:{count}" for shape, count in shape_counts.most_common())
        examples: list[str] = []
        if contexts is not None and len(contexts) == len(values):
            for shape, _ in shape_counts.most_common():
                picked = 0
                for arr, rec in zip(values, contexts, strict=False):
                    if _shape_of_array(arr) != shape:
                        continue
                    examples.append(f"shape={shape} <- {_format_record_context(rec)}")
                    picked += 1
                    if picked >= 2:
                        break
                if len(examples) >= 10:
                    break
        example_text = " | ".join(examples) if examples else "n/a"
        raise ValueError(
            "all input arrays must have the same shape; "
            f"feature={feature_key}, metric={metric}, shape_counts=[{shape_summary}], "
            f"examples=[{example_text}]"
        ) from exc


def _build_episode_file_resolver(episodes_stats_path: Path):
    """Return resolver(episode_index)->expected parquet path when info.json is available."""

    info_path = episodes_stats_path.parent / "info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to parse info.json ({info_path}): {exc}")
        return None

    data_path_template = info.get("data_path")
    chunks_size = info.get("chunks_size")
    if not isinstance(data_path_template, str):
        return None
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        chunks_size = None

    dataset_root = episodes_stats_path.parent.parent

    def _resolve(episode_index: int | None) -> str | None:
        if episode_index is None:
            return None
        fmt_values = {"episode_index": int(episode_index)}
        if chunks_size is not None:
            fmt_values["episode_chunk"] = int(episode_index) // chunks_size
        try:
            rel_path = data_path_template.format(**fmt_values)
        except Exception:  # noqa: BLE001
            return None
        return str(dataset_root / rel_path)

    return _resolve

def _required_feature_keys(action_mode: str, action_column: str) -> set[str]:
    """Return minimal feature keys required to produce final state/actions norm stats."""
    keys = {
        "observation.state",
        "action.relative",
    }
    if action_mode == "absolute_arm_head_relative_gripper_base":
        keys.add("action.absolute")
    elif action_mode == "state_diff_arm_head_relative_gripper_base":
        keys.add("action.state_diff")
    if action_column:
        keys.add(action_column)
    return keys

def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def aggregate_feature_stats(
    feature_key: str,
    stats_ft_list: list[dict[str, np.ndarray]],
    *,
    contexts: list[EpisodeStatsRecord] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = _stack_with_debug(
        [s["mean"] for s in stats_ft_list],
        feature_key=feature_key,
        metric="mean",
        contexts=contexts,
    )
    variances = _stack_with_debug(
        [s["std"] ** 2 for s in stats_ft_list],
        feature_key=feature_key,
        metric="std",
        contexts=contexts,
    )
    counts = _stack_with_debug(
        [s["count"] for s in stats_ft_list],
        feature_key=feature_key,
        metric="count",
        contexts=contexts,
    )
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(
            _stack_with_debug(
                [s["min"] for s in stats_ft_list],
                feature_key=feature_key,
                metric="min",
                contexts=contexts,
            ),
            axis=0,
        ),
        "max": np.max(
            _stack_with_debug(
                [s["max"] for s in stats_ft_list],
                feature_key=feature_key,
                metric="max",
                contexts=contexts,
            ),
            axis=0,
        ),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_records: list[EpisodeStatsRecord]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """
    stats_list = [rec.stats for rec in stats_records]
    _assert_type_and_shape(stats_list)

    data_keys = {key for rec in stats_records for key in rec.stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        entries = [(rec.stats[key], rec) for rec in stats_records if key in rec.stats]
        stats_with_key = [stats for stats, _ in entries]
        contexts = [rec for _, rec in entries]
        aggregated_stats[key] = aggregate_feature_stats(key, stats_with_key, contexts=contexts)

    return aggregated_stats

# --- HSR padding/transform helpers (参照: src/openpi/policies/hsr_policy.py) ---

def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)



def _approx_transform_mean_std(mean: float, std: float, fn) -> tuple[float, float]:
    """非線形変換後の(mean,std)を簡易推定。
    中心差分で 0.5*(f(m+std)-f(m-std)) を新しいstd近似とする。
    """
    if std == 0:
        return fn(mean), 0.0
    upper = fn(mean + std)
    lower = fn(mean - std)
    return fn(mean), abs(upper - lower) / 2.0

def pad_hsr_state_stats(stat: dict, adapt_to_pi: bool, target_dim: int = 32) -> dict:
    """8次元(observation.state)統計 -> (14 -> target_dim)に拡張し、norm_stats形式に合わせる。

    Mapping (hsr_policy._decode_state): aligned_ids = [0,1,2,3,4,6,11,12]
    (index6 が gripper, 11/12 が head joints)。
    """
    aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12]
    out = {k: [0.0] * target_dim for k in stat.keys() if k in ("min", "max", "mean", "std")}
    # counts は総フレーム数
    count = stat.get("count", [0])[0]
    # 入力配列
    for name in ["min", "max", "mean", "std"]:
        if name not in stat:
            continue
        arr = np.asarray(stat[name])  # shape (8,)
        for src_i, dst_i in enumerate(aligned_ids):
            val = float(arr[src_i])
            out[name][dst_i] = val

    # if adapt_to_pi and "mean" in out:
    #     # gripper index 6 について非線形変換を apply (mean,std,min,max)
    #     g_idx_in_aligned = aligned_ids.index(6)  # position in original 8-dim -> 6
    #     orig_mean = float(stat["mean"][g_idx_in_aligned])
    #     orig_std = float(stat["std"][g_idx_in_aligned]) if "std" in stat else 0.0
    #     new_mean, new_std = _approx_transform_mean_std(orig_mean, orig_std, _gripper_to_angular)
    #     out["mean"][6] = new_mean
    #     if "std" in out:
    #         out["std"][6] = new_std
    #     # min/max 単純変換
    #     if "min" in out:
    #         out["min"][6] = _gripper_to_angular(float(stat["min"][g_idx_in_aligned]))
    #     if "max" in out:
    #         out["max"][6] = _gripper_to_angular(float(stat["max"][g_idx_in_aligned]))

    return {
    "min": out.get("min"),
    "max": out.get("max"),
    "mean": out.get("mean"),
    "std": out.get("std"),
    "q01": None,  # 後段で計算（正規近似）
    "q99": None,
    "count": [count],
    }

def pad_hsr_action_stats(stat: dict, adapt_to_pi: bool, target_dim: int = 32) -> dict:
    """11次元(action.relative) -> (16 -> target_dim)。
    Mapping (hsr_policy._decode_actions): aligned_ids = [0,1,2,3,4,6,11,12,13,14,15]
    ここでは非線形変換は行わない (hsr_policyでも未適用)。
    """
    aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]
    out = {k: [0.0] * target_dim for k in stat.keys() if k in ("min", "max", "mean", "std")}
    count = stat.get("count", [0])[0]
    for name in ["min", "max", "mean", "std"]:
        if name not in stat:
            continue
        arr = np.asarray(stat[name])  # shape (11,)
        for src_i, dst_i in enumerate(aligned_ids):
            out[name][dst_i] = float(arr[src_i])
    return {
    "min": out.get("min"),
    "max": out.get("max"),
    "mean": out.get("mean"),
    "std": out.get("std"),
    "q01": None,  # 後段で計算
    "q99": None,
    "count": [count],
    }

def combine_action_stats(state_diff: dict | None, relative: dict, base_dim: int = 3) -> dict:
    """Combine action.state_diff with base dims from action.relative."""
    if state_diff is None:
        return relative

    state_dim = len(state_diff.get("mean", []))
    relative_dim = len(relative.get("mean", []))
    if state_dim == relative_dim:
        return state_diff
    if state_dim + base_dim != relative_dim:
        raise ValueError(
            f"Cannot combine action stats: state_diff dim={state_dim}, relative dim={relative_dim}, base_dim={base_dim}"
        )

    combined = {}
    for key in ("min", "max", "mean", "std"):
        if key in state_diff and key in relative:
            combined[key] = np.concatenate(
                [np.asarray(state_diff[key]), np.asarray(relative[key])[-base_dim:]], axis=0
            )
    combined["count"] = state_diff.get("count", relative.get("count"))
    return combined

def combine_action_stats_arm_head_relative_gripper_base(
    state_diff: dict | None,
    relative: dict,
    *,
    arm_dim: int = 5,
    gripper_index: int = 5,
    head_dim: int = 2,
    base_dim: int = 3,
) -> dict:
    """Combine arm/head from state_diff with gripper/base from relative."""
    if state_diff is None:
        return relative

    state_dim = len(state_diff.get("mean", []))
    relative_dim = len(relative.get("mean", []))
    expected_state_dim = arm_dim + 1 + head_dim
    expected_relative_dim = expected_state_dim + base_dim
    if state_dim != expected_state_dim or relative_dim != expected_relative_dim:
        raise ValueError(
            "Cannot combine action stats: "
            f"state_diff dim={state_dim}, relative dim={relative_dim}, "
            f"expected_state_dim={expected_state_dim}, expected_relative_dim={expected_relative_dim}"
        )

    combined = {}
    for key in ("min", "max", "mean", "std"):
        if key in state_diff and key in relative:
            left = np.asarray(state_diff[key])[:arm_dim]
            gripper = np.asarray(relative[key])[gripper_index : gripper_index + 1]
            head_start = gripper_index + 1
            head = np.asarray(state_diff[key])[head_start : head_start + head_dim]
            base = np.asarray(relative[key])[-base_dim:]
            combined[key] = np.concatenate([left, gripper, head, base], axis=0)
    combined["count"] = state_diff.get("count", relative.get("count"))
    return combined

def load_episodes_stats(path: str, *, include_features: set[str] | None = None) -> list[EpisodeStatsRecord]:
    stats_list: list[EpisodeStatsRecord] = []
    skipped_stats_lines = 0
    skipped_feature_stats = 0
    skipped_filtered_features = 0
    path_obj = Path(path)
    resolve_episode_file = _build_episode_file_resolver(path_obj)
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("stats") is None:
                skipped_stats_lines += 1
                continue
            conv = {}
            for fkey, sval in obj["stats"].items():
                if include_features is not None and fkey not in include_features:
                    skipped_filtered_features += 1
                    continue
                if sval is None:
                    skipped_feature_stats += 1
                    continue
                conv[fkey] = {k: np.asarray(v) for k, v in sval.items()}
            episode_index = obj.get("episode_index")
            if isinstance(episode_index, bool):
                episode_index = None
            elif episode_index is not None:
                try:
                    episode_index = int(episode_index)
                except (TypeError, ValueError):
                    episode_index = None
            episode_file = resolve_episode_file(episode_index) if resolve_episode_file else None
            stats_list.append(
                EpisodeStatsRecord(
                    line_no=line_no,
                    episode_index=episode_index,
                    episode_file=episode_file,
                    stats=conv,
                )
            )
    if not stats_list:
        raise ValueError("No stats found in episodes_stats file")
    if skipped_stats_lines or skipped_feature_stats:
        print(
            "[WARN] Skipped null stats entries: "
            f"lines={skipped_stats_lines}, features={skipped_feature_stats}"
        )
    if include_features is not None:
        print(
            "[INFO] Feature filtering enabled: "
            f"kept={sorted(include_features)}, filtered_features={skipped_filtered_features}"
        )
    return stats_list

def load_gripper_data_from_parquet_dir(parquet_dir: str, state_column: str = "observation.state", action_column: str = "action.relative") -> tuple[list[np.ndarray], list[np.ndarray]]:
    """parquetディレクトリから各エピソードのgripperデータを読み込み、エピソード毎のgripper値リストを返す。
    
    Args:
        parquet_dir: parquetファイルが格納されているディレクトリのパス
        state_column: state gripper値が含まれるカラム名（デフォルト: "observation.state"）
        action_column: action gripper値が含まれるカラム名（デフォルト: "action.relative"）
    
    Returns:
        (state_gripper_data, action_gripper_data): 各エピソードのstate/action gripper値（index 5）のリスト
    """
    parquet_dir_path = Path(parquet_dir)
    if not parquet_dir_path.exists():
        raise ValueError(f"ディレクトリが存在しません: {parquet_dir}")
    
    # episode_*.parquetファイルを検索
    parquet_files = sorted(glob.glob(str(parquet_dir_path / "episode_*.parquet")))
    if not parquet_files:
        raise ValueError(f"parquetファイルが見つかりません: {parquet_dir}")
    
    print(f"Found {len(parquet_files)} parquet files in {parquet_dir}")
    
    state_gripper_data_per_episode = []
    action_gripper_data_per_episode = []
    
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            
            # State gripper data
            state_gripper_values = []
            if state_column in df.columns:
                for _, row in df.iterrows():
                    state_data = row[state_column]
                    if isinstance(state_data, (list, np.ndarray)) and len(state_data) > 5:
                        state_gripper_values.append(float(state_data[5]))  # index 5がgripper
                    elif hasattr(state_data, '__getitem__') and len(state_data) > 5:
                        state_gripper_values.append(float(state_data[5]))
            else:
                print(f"Warning: '{state_column}'カラムが見つかりません in {parquet_file}")
            
            # Action gripper data
            action_gripper_values = []
            if action_column in df.columns:
                for _, row in df.iterrows():
                    action_data = row[action_column]
                    if isinstance(action_data, (list, np.ndarray)) and len(action_data) > 5:
                        gripper_value = float(action_data[5])  # index 5がgripper
                        action_gripper_values.append(gripper_value)
                    elif hasattr(action_data, '__getitem__') and len(action_data) > 5:
                        gripper_value = float(action_data[5])
                        action_gripper_values.append(gripper_value)
                
                # # デバッグ: action.relativeの最初の数行を確認
                # if len(action_gripper_values) > 0:
                #     print(f"  Debug: First few action.relative[5] values: {action_gripper_values[:5]}")
                #     print(f"  Debug: action.relative[5] range: [{min(action_gripper_values):.6f}, {max(action_gripper_values):.6f}]")
            else:
                print(f"Warning: '{action_column}'カラムが見つかりません in {parquet_file}")
            
            if state_gripper_values:
                state_gripper_data_per_episode.append(np.array(state_gripper_values))
            if action_gripper_values:
                action_gripper_data_per_episode.append(np.array(action_gripper_values))
                
        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue
    
    print(f"Successfully loaded state gripper data from {len(state_gripper_data_per_episode)} episodes in {parquet_dir}")
    print(f"Successfully loaded action gripper data from {len(action_gripper_data_per_episode)} episodes in {parquet_dir}")
    return state_gripper_data_per_episode, action_gripper_data_per_episode

def load_gripper_data_from_multiple_dirs(parquet_dirs: list[str], state_column: str = "observation.state", action_column: str = "action.relative") -> tuple[list[np.ndarray], list[np.ndarray]]:
    """複数のparquetディレクトリから各エピソードのgripperデータを読み込み、エピソード毎のgripper値リストを返す。
    
    Args:
        parquet_dirs: parquetファイルが格納されているディレクトリのパスのリスト
        state_column: state gripper値が含まれるカラム名（デフォルト: "observation.state"）
        action_column: action gripper値が含まれるカラム名（デフォルト: "action.relative"）
    
    Returns:
        (state_gripper_data, action_gripper_data): 各エピソードのstate/action gripper値（index 5）のリスト（全ディレクトリ統合）
    """
    all_state_gripper_data = []
    all_action_gripper_data = []
    
    for parquet_dir in parquet_dirs:
        print(f"Processing directory: {parquet_dir}")
        try:
            state_gripper_data, action_gripper_data = load_gripper_data_from_parquet_dir(parquet_dir, state_column, action_column)
            all_state_gripper_data.extend(state_gripper_data)
            all_action_gripper_data.extend(action_gripper_data)
        except Exception as e:
            print(f"Error processing directory {parquet_dir}: {e}")
            continue
    
    print(f"Total state gripper episodes loaded from {len(parquet_dirs)} directories: {len(all_state_gripper_data)}")
    print(f"Total action gripper episodes loaded from {len(parquet_dirs)} directories: {len(all_action_gripper_data)}")
    return all_state_gripper_data, all_action_gripper_data

def compute_gripper_stats_with_transform(
    gripper_data_per_episode: list[np.ndarray],
    *,
    convert_gripper: bool,
) -> dict:
    """gripper データに非線形変換を適用してstatsを計算する。
    compute_norm_stats.pyのRunningStatsと同様の方法で統計を計算。
    
    Args:
        gripper_data_per_episode: エピソード毎のgripper値のリスト
    
    Returns:
        変換後のgripper統計情報 {min, max, mean, std, count}
    """
    if not gripper_data_per_episode:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    
    print(f"Computing gripper stats for {len(gripper_data_per_episode)} episodes")
    
    # 各エピソードのgripperデータに必要なら非線形変換を適用してから統計を計算
    all_transformed_data = []

    for episode_data in gripper_data_per_episode:
        if len(episode_data) > 0:
            if convert_gripper:
                transformed_episode = _gripper_to_angular(episode_data)
            else:
                transformed_episode = episode_data
            all_transformed_data.append(transformed_episode)
    
    if not all_transformed_data:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    
    # 全エピソードの変換後データを結合
    all_gripper_transformed = np.concatenate(all_transformed_data)
    
    # 統計を計算（compute_norm_stats.pyと同様）
    stats = {
        "min": [float(np.min(all_gripper_transformed))],
        "max": [float(np.max(all_gripper_transformed))],
        "mean": [float(np.mean(all_gripper_transformed))],
        "std": [float(np.std(all_gripper_transformed, ddof=0))],  # ddof=0 for population std
        "count": [len(all_gripper_transformed)]
    }
    
    print(f"Gripper stats computed: count={stats['count'][0]}, "
          f"mean={stats['mean'][0]:.6f}, std={stats['std'][0]:.6f}, "
          f"min={stats['min'][0]:.6f}, max={stats['max'][0]:.6f}")
    
    return stats

def compute_gripper_stats_with_action_transform(
    gripper_data_per_episode: list[np.ndarray],
    *,
    convert_gripper: bool,
) -> dict:
    """actionのgripper データに_gripper_from_angular_inv変換を適用してstatsを計算する。
    hsr_policy.py の _encode_actions_inv() に合わせた処理。
    
    Args:
        gripper_data_per_episode: エピソード毎のgripper値のリスト
    
    Returns:
        _gripper_from_angular_inv変換後のgripper統計情報 {min, max, mean, std, count}
    """
    if not gripper_data_per_episode:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    
    print(f"Computing action gripper stats with _gripper_from_angular_inv transform for {len(gripper_data_per_episode)} episodes")
    
    # 各エピソードのgripperデータに必要なら_gripper_from_angular_inv変換を適用してから統計を計算
    all_transformed_data = []

    for episode_data in gripper_data_per_episode:
        if len(episode_data) > 0:
            if convert_gripper:
                transformed_episode = _gripper_from_angular_inv(episode_data)
            else:
                transformed_episode = episode_data
            all_transformed_data.append(transformed_episode)
    
    if not all_transformed_data:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    
    # 全エピソードの変換後データを結合
    all_gripper_transformed = np.concatenate(all_transformed_data)
    
    # 統計を計算（compute_norm_stats.pyと同様）
    stats = {
        "min": [float(np.min(all_gripper_transformed))],
        "max": [float(np.max(all_gripper_transformed))],
        "mean": [float(np.mean(all_gripper_transformed))],
        "std": [float(np.std(all_gripper_transformed, ddof=0))],  # ddof=0 for population std
        "count": [len(all_gripper_transformed)]
    }
    
    print(f"Action gripper stats with _gripper_from_angular_inv computed: count={stats['count'][0]}, "
          f"mean={stats['mean'][0]:.6f}, std={stats['std'][0]:.6f}, "
          f"min={stats['min'][0]:.6f}, max={stats['max'][0]:.6f}")
    
    return stats

def numpy_dict_to_lists(d: dict):
    out = {}
    for fkey, stat in d.items():
        out[fkey] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stat.items()}
    return out

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
    ap.add_argument("--chunk-dir", required=True, help="Path(s) to data directory(ies) containing chunk* directories containing episode parquet files (can specify multiple directories)")
    ap.add_argument("--state-column", default="observation.state", help="Column name containing state gripper data (default: observation.state)")
    ap.add_argument("--action-column", default="action.relative", help="Column name containing action gripper data (default: action.relative)")
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
    args = ap.parse_args()

    include_features = None
    if not args.include_all_features:
        include_features = _required_feature_keys(args.action_mode, args.action_column)
    stats_records = load_episodes_stats(args.episodes_stats, include_features=include_features)

    if args.convert_gripper:
        # gripper部分をchunkディレクトリから読み込んで非線形変換を適用
        print(f"Loading gripper data from chunk directories: {args.chunk_dir}")
        # chunk_dirからchunk-*ディレクトリを自動検出
        chunk_dirs = []
        data_dir = args.chunk_dir
        data_path = Path(data_dir)
        if data_path.exists():
            # chunk-*パターンのディレクトリを検索
            found_chunks = sorted(glob.glob(str(data_path / "chunk-*")))
            if found_chunks:
                chunk_dirs.extend(found_chunks)
                print(
                    f"Found {len(found_chunks)} chunk directories in {data_dir}: "
                    f"{[Path(p).name for p in found_chunks]}"
                )
            else:
                print(f"No chunk-* directories found in {data_dir}")
        else:
            print(f"Warning: Data directory does not exist: {data_dir}")

        if chunk_dirs:
            state_gripper_data, action_gripper_data = load_gripper_data_from_multiple_dirs(
                chunk_dirs, args.state_column, args.action_column
            )
            # import pdb; pdb.set_trace()
            # State gripper統計を計算
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

            # Action gripper統計を計算（_gripper_from_angular_inv変換 - hsr_policy.py の _encode_actions_inv に合わせる）
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
    
    # gripper統計が取得できた場合のみ更新
    if state_gripper_stats or action_gripper_stats:
        for i, rec in enumerate(stats_records):
            ep_stats = rec.stats
            # State gripper統計を更新
            if state_gripper_stats and "observation.state" in ep_stats:
                state_stats = ep_stats["observation.state"]
                # gripper統計をindex 5に設定
                for key in ["min", "max", "mean", "std"]:
                    if key in state_stats and len(state_stats[key]) > 5:
                        # 全エピソード共通のgripper統計を使用
                        state_stats[key][5] = state_gripper_stats[key][0]
            
            # Action gripper統計を更新
            if action_gripper_stats and args.action_column in ep_stats:
                action_stats = ep_stats[args.action_column]
                # gripper統計をindex 5に設定
                for key in ["min", "max", "mean", "std"]:
                    if key in action_stats and len(action_stats[key]) > 5:
                        # 全エピソード共通のgripper統計を使用
                        action_stats[key][5] = action_gripper_stats[key][0]
    else:
        print("Gripper stats not updated (no chunk directories processed)")

    agg = aggregate_stats(stats_records)
    agg_lists = numpy_dict_to_lists(agg)

    adapt = not args.no_adapt_to_pi
    adapt_state = adapt # and not args.exact_gripper  # exact後は二重変換回避
    # 必要キー存在チェック
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
    
    from math import erf as _m_erf, sqrt

    def _std_normal_cdf(x) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        # math.erf はスカラーのみ対応なので np.vectorize
        v_erf = np.vectorize(_m_erf)
        return 0.5 * (1.0 + v_erf(x_arr / sqrt(2.0)))

    def _mixture_quantiles(
        per_episode: list[EpisodeStatsRecord],
        feature_key: str,
        q_levels=(0.01, 0.99),
    ) -> dict[float, list[float]]:
        # 期待: 各エピソード stats_records[i].stats[feature_key] に required keys
        comps: list[tuple[dict[str, np.ndarray], EpisodeStatsRecord]] = []
        for rec in per_episode:
            ep_stats = rec.stats
            if feature_key not in ep_stats:
                continue
            s = ep_stats[feature_key]
            if not all(k in s for k in ("mean", "std", "min", "max", "count")):
                continue
            comps.append((s, rec))
        if not comps:
            return {q: [] for q in q_levels}
        comp_stats = [c for c, _ in comps]
        comp_contexts = [rec for _, rec in comps]
        mean_arr = _stack_with_debug(
            [c["mean"] for c in comp_stats],
            feature_key=feature_key,
            metric="mean@quantile",
            contexts=comp_contexts,
        )
        std_arr = _stack_with_debug(
            [c["std"] for c in comp_stats],
            feature_key=feature_key,
            metric="std@quantile",
            contexts=comp_contexts,
        )
        min_arr = _stack_with_debug(
            [c["min"] for c in comp_stats],
            feature_key=feature_key,
            metric="min@quantile",
            contexts=comp_contexts,
        )
        max_arr = _stack_with_debug(
            [c["max"] for c in comp_stats],
            feature_key=feature_key,
            metric="max@quantile",
            contexts=comp_contexts,
        )
        count_arr = _stack_with_debug(
            [c["count"] for c in comp_stats],
            feature_key=feature_key,
            metric="count@quantile",
            contexts=comp_contexts,
        )  # shape (E,1) or (E,)
        if count_arr.ndim > 1:
            count_arr = count_arr.reshape(count_arr.shape[0], -1)[:, 0]
        total_counts = count_arr.sum()
        weights = count_arr / (total_counts if total_counts > 0 else 1)
        dim = mean_arr.shape[1]
        results = {q: [0.0] * dim for q in q_levels}
        # precompute global support per dim
        global_min = np.min(min_arr, axis=0)
        global_max = np.max(max_arr, axis=0)
        # 防御: min==max の次元 → その値
        for d in range(dim):
            if not np.isfinite(global_min[d]) or not np.isfinite(global_max[d]):
                for q in q_levels:
                    results[q][d] = float('nan')
                continue
            if global_min[d] >= global_max[d]:  # 定数
                val = float(global_min[d])
                for q in q_levels:
                    results[q][d] = val
                continue
            mu_j = mean_arr[:, d]
            sigma_j = std_arr[:, d]
            lo_j = min_arr[:, d]
            hi_j = max_arr[:, d]
            # 事前計算 (有効な分布成分)
            # sigma_j <= 0 (あるいは非常に小さい) は点質量扱い
            point_mask = (sigma_j <= 1e-12) | (hi_j <= lo_j)
            cont_mask = ~point_mask
            mu_c = mu_j[cont_mask]
            sig_c = sigma_j[cont_mask]
            lo_c = lo_j[cont_mask]
            hi_c = hi_j[cont_mask]
            w_point = weights[point_mask]
            w_cont = weights[cont_mask]
            # Truncated 正規用の a,b, Phi(a), Phi(b), denom
            if mu_c.size > 0:
                a = (lo_c - mu_c) / sig_c
                b = (hi_c - mu_c) / sig_c
                Phi_a = _std_normal_cdf(a)
                Phi_b = _std_normal_cdf(b)
                denom = np.clip(Phi_b - Phi_a, 1e-12, None)

            def cdf_scalar(x: float) -> float:
                # 点質量成分
                total = float(np.sum(w_point * (x >= mu_j[point_mask]))) if w_point.size else 0.0
                # 連続成分
                if mu_c.size:
                    z = (x - mu_c) / sig_c
                    Phi_z = _std_normal_cdf(z)
                    p = np.clip((Phi_z - Phi_a) / denom, 0.0, 1.0)
                    # 下端/上端より外側を明示的に補正
                    p = np.where(x <= lo_c, 0.0, p)
                    p = np.where(x >= hi_c, 1.0, p)
                    total += float(np.sum(w_cont * p))
                return total
            # 量子化関数（CDF）
            # 各 q を二分探索
            for q in q_levels:
                left, right = global_min[d], global_max[d]
                for _ in range(50):  # 高精度 (約1e-15*range)
                    mid = 0.5 * (left + right)
                    if cdf_scalar(mid) < q:
                        left = mid
                    else:
                        right = mid
                results[q][d] = float(0.5 * (left + right))
        return results

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

    # 生 state / actions の q01,q99 を計算
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

    # パディング済み配列へマッピング (state: aligned_ids, actions: aligned_ids)
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

    # 最終キー整形 (mean,std,q01,q99 のみ残す)
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

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_file).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default))
    print(f"[OK] wrote {args.output_file}")

if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
