import argparse
import csv
import heapq
import os
from pathlib import Path
from typing import Any

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


def _get_first_present(data: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _try_get_column(dataset: Any, key: str) -> Any | None:
    for attr in ("hf_dataset", "_hf_dataset", "dataset", "_dataset", "data"):
        obj = getattr(dataset, attr, None)
        if obj is None:
            continue
        try:
            return obj[key]
        except Exception:
            continue
    return None


def _try_get_column_names(dataset: Any) -> list[str] | None:
    for attr in ("hf_dataset", "_hf_dataset", "dataset", "_dataset", "data"):
        obj = getattr(dataset, attr, None)
        if obj is None:
            continue
        names = getattr(obj, "column_names", None)
        if names is not None:
            return list(names)
    return None


def _apply_transforms(data: dict[str, Any], transforms: list[Any]) -> dict[str, Any]:
    for transform in transforms:
        data = transform(data)
    return data


def _print_action_norm_stats(norm_stats: dict[str, Any] | None, dim: int) -> None:
    if norm_stats is None or "actions" not in norm_stats:
        print("Norm stats: actions not found or stats missing.")
        return
    stats = norm_stats["actions"]
    mean = stats.mean[dim] if stats.mean is not None and dim < len(stats.mean) else None
    std = stats.std[dim] if stats.std is not None and dim < len(stats.std) else None
    q01 = stats.q01[dim] if stats.q01 is not None and dim < len(stats.q01) else None
    q99 = stats.q99[dim] if stats.q99 is not None and dim < len(stats.q99) else None
    print(f"Norm stats for actions dim {dim}: mean={mean} std={std} q01={q01} q99={q99}")


def _infer_action_key(data_config: Any) -> str:
    action_mode = getattr(data_config, "action_mode", None)
    if isinstance(action_mode, str):
        if "state_diff" in action_mode:
            return "action.state_diff"
        if "relative" in action_mode:
            return "action.relative"
    return "action.relative"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan LeRobot dataset for action outliers.")
    parser.add_argument("config_name", help="Training config name (e.g., pi05_hsr_optimal_microwave).")
    parser.add_argument(
        "--action-key",
        default="action.state_diff",
        help="Action key to inspect in the raw dataset (default: action.state_diff).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=6,
        help="Action dimension index to inspect (0-based). Default 6 (head_pan for HSR state_diff).",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of top outliers to report.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to scan (default: all).",
    )
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=None,
        help="Stop early when abs max exceeds this threshold.",
    )
    parser.add_argument(
        "--compare-normalize",
        action="store_true",
        help="Compare actions before/after normalization (uses transformed pipeline).",
    )
    parser.add_argument(
        "--fast-column",
        action="store_true",
        help="Try to read the action column directly from the underlying HF dataset (raw scan only).",
    )
    parser.add_argument(
        "--parquet-fast",
        action="store_true",
        help="Read action column directly from parquet files for fast raw scan (requires pyarrow).",
    )
    parser.add_argument(
        "--scan-invalid",
        action="store_true",
        help="Scan all parquet files for invalid action values and write a CSV report.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e6,
        help="Absolute value threshold for invalid action values (used with --scan-invalid).",
    )
    parser.add_argument(
        "--output-csv",
        default="invalid_actions.csv",
        help="Output CSV path for invalid scan results.",
    )
    parser.add_argument(
        "--exclude-episodes-csv",
        default=None,
        help="Output CSV path listing unique episode_index values to exclude.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Base directory for lerobot datasets (defaults to DATA_DIR env or /home/datasets/lerobot_datasets).",
    )
    parser.add_argument(
        "--print-keys",
        action="store_true",
        help="Print dataset item keys from the first sample for debugging.",
    )
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if args.scan_invalid:
        args.action_key = _infer_action_key(data_config)
    _print_action_norm_stats(data_config.norm_stats, args.dim)

    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )

    total = len(dataset)
    if args.limit is not None:
        total = min(total, args.limit)

    top_k = max(args.top_k, 1)
    results: list[tuple[float, int, int | None, int | None]] = []
    compare_results: list[tuple[float, float, float, int, int | None, int | None]] = []

    if args.scan_invalid:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "pyarrow is required for --scan-invalid. Install it (e.g., `uv pip install pyarrow`)."
            ) from exc

        data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "/home/datasets/lerobot_datasets"))
        dataset_name = data_config.repo_id.split("/")[-1]
        parquet_root = data_dir / dataset_name / "data"
        if not parquet_root.exists():
            raise FileNotFoundError(f"Parquet root not found: {parquet_root}")

        parquet_files = sorted(parquet_root.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {parquet_root}")

        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exclude_path = Path(args.exclude_episodes_csv) if args.exclude_episodes_csv else None
        exclude_eps: set[int] = set()
        print(f"Scanning {len(parquet_files)} parquet files under: {parquet_root}")
        print(f"Action key: {args.action_key}, threshold: {args.threshold}")
        print(f"Writing CSV: {output_path}")
        if exclude_path is not None:
            exclude_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing exclude episodes CSV: {exclude_path}")

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode_index",
                    "frame_index",
                    "parquet_file",
                    "action_key",
                    "dim",
                    "value",
                    "reason",
                ]
            )

            for path in parquet_files:
                cols = [args.action_key, "episode_index", "frame_index"]
                table = pq.read_table(path, columns=cols)
                actions = table.column(0).to_pylist()
                episode_indices = table.column(1).to_pylist()
                frame_indices = table.column(2).to_pylist()

                for i, row in enumerate(actions):
                    if row is None:
                        continue
                    action = np.asarray(row)
                    if action.ndim == 1:
                        action = action[None, :]
                    if action.shape[-1] == 0:
                        continue
                    values = action.reshape(-1, action.shape[-1])
                    for dim in range(values.shape[-1]):
                        col = values[:, dim]
                        if np.isnan(col).any():
                            if episode_indices[i] is not None:
                                exclude_eps.add(int(episode_indices[i]))
                            writer.writerow(
                                [
                                    int(episode_indices[i]) if episode_indices[i] is not None else "",
                                    int(frame_indices[i]) if frame_indices[i] is not None else "",
                                    path.name,
                                    args.action_key,
                                    dim,
                                    "nan",
                                    "nan",
                                ]
                            )
                        if np.isinf(col).any():
                            if episode_indices[i] is not None:
                                exclude_eps.add(int(episode_indices[i]))
                            writer.writerow(
                                [
                                    int(episode_indices[i]) if episode_indices[i] is not None else "",
                                    int(frame_indices[i]) if frame_indices[i] is not None else "",
                                    path.name,
                                    args.action_key,
                                    dim,
                                    "inf",
                                    "inf",
                                ]
                            )
                        max_abs = float(np.max(np.abs(col)))
                        if max_abs > args.threshold:
                            if episode_indices[i] is not None:
                                exclude_eps.add(int(episode_indices[i]))
                            max_val = float(col[np.argmax(np.abs(col))])
                            writer.writerow(
                                [
                                    int(episode_indices[i]) if episode_indices[i] is not None else "",
                                    int(frame_indices[i]) if frame_indices[i] is not None else "",
                                    path.name,
                                    args.action_key,
                                    dim,
                                    f"{max_val:.6g}",
                                    f"abs_gt_{args.threshold:g}",
                                ]
                            )
        if exclude_path is not None:
            with exclude_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode_index"])
                for ep in sorted(exclude_eps):
                    writer.writerow([ep])
        print("Scan complete.")
        return

    if args.parquet_fast and not args.compare_normalize:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "pyarrow is required for --parquet-fast. Install it (e.g., `uv pip install pyarrow`)."
            ) from exc

        data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "/home/datasets/lerobot_datasets"))
        dataset_name = data_config.repo_id.split("/")[-1]
        parquet_root = data_dir / dataset_name / "data"
        if not parquet_root.exists():
            raise FileNotFoundError(f"Parquet root not found: {parquet_root}")

        parquet_files = sorted(parquet_root.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {parquet_root}")

        print(f"Using parquet fast scan under: {parquet_root}")
        sample_count = 0
        heap: list[tuple[float, int, str]] = []

        for path in parquet_files:
            table = pq.read_table(path, columns=[args.action_key])
            column = table.column(0)
            for row in column.to_pylist():
                if row is None:
                    continue
                action = np.asarray(row)
                if action.ndim == 1:
                    action = action[None, :]
                if action.shape[-1] <= args.dim:
                    raise ValueError(
                        f"Action dim {args.dim} out of range for shape {action.shape}. "
                        f"Check '--dim' or '--action-key'."
                    )
                values = action[..., args.dim]
                absmax = float(np.max(np.abs(values)))
                if len(heap) < top_k:
                    heapq.heappush(heap, (absmax, sample_count, path.name))
                elif absmax > heap[0][0]:
                    heapq.heapreplace(heap, (absmax, sample_count, path.name))
                if args.stop_threshold is not None and absmax > args.stop_threshold:
                    print("Threshold exceeded in raw parquet scan:")
                    print(f"absmax={absmax:.6g} sample_idx={sample_count} file={path.name}")
                    return
                sample_count += 1
                if args.limit is not None and sample_count >= args.limit:
                    break
            if args.limit is not None and sample_count >= args.limit:
                break

        results = sorted(heap, key=lambda x: x[0], reverse=True)
        print(f"Scanned {sample_count} samples from {data_config.repo_id}")
        print(f"Action key: {args.action_key}, dim: {args.dim}")
        print("Top outliers (abs max):")
        for rank, (value, idx, fname) in enumerate(results, start=1):
            print(f"{rank:02d} absmax={value:.6g} sample_idx={idx} file={fname}")
        return

    if args.fast_column and not args.compare_normalize:
        if args.print_keys:
            sample = dataset[0]
            print("Sample item keys:", sorted(sample.keys()))
        column = _try_get_column(dataset, args.action_key)
        if column is None:
            names = _try_get_column_names(dataset)
            if names:
                print("Available columns:", names)
            print(
                "Fast column access is not available for this dataset/key. "
                "Falling back to regular scan (this may be slow)."
            )
        else:
            total = min(total, len(column))
            print("Using fast column access for raw scan.")
            for idx in range(total):
                action = np.asarray(column[idx])
                if action.ndim == 1:
                    action = action[None, :]
                if action.shape[-1] <= args.dim:
                    raise ValueError(
                        f"Action dim {args.dim} out of range for shape {action.shape}. "
                        f"Check '--dim' or '--action-key'."
                    )
                values = action[..., args.dim]
                absmax = float(np.max(np.abs(values)))
                results.append((absmax, idx, None, None))
                if args.stop_threshold is not None and absmax > args.stop_threshold:
                    print("Threshold exceeded in raw scan:")
                    print(f"absmax={absmax:.6g} idx={idx}")
                    return

            results.sort(key=lambda x: x[0], reverse=True)
            print(f"Scanned {total} samples from {data_config.repo_id}")
            print(f"Action key: {args.action_key}, dim: {args.dim}")
            print("Top outliers (abs max):")
            for rank, (value, idx, _, _) in enumerate(results[:top_k], start=1):
                print(f"{rank:02d} absmax={value:.6g} idx={idx}")
            return

    for idx in range(total):
        item = dataset[idx]
        if idx == 0 and args.print_keys:
            print("First item keys:", sorted(item.keys()))

        if args.compare_normalize:
            raw = dict(item)
            pre = _apply_transforms(
                raw,
                [
                    *data_config.repack_transforms.inputs,
                    *data_config.data_transforms.inputs,
                ],
            )
            if "actions" not in pre:
                raise KeyError("Actions not found after repack/data transforms.")
            pre_actions = np.asarray(pre["actions"])
            if pre_actions.ndim == 1:
                pre_actions = pre_actions[None, :]
            if pre_actions.shape[-1] <= args.dim:
                raise ValueError(
                    f"Action dim {args.dim} out of range for shape {pre_actions.shape}. "
                    f"Check '--dim'."
                )

            normalized = _apply_transforms(
                pre,
                [
                    _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                ],
            )
            if "actions" not in normalized:
                raise KeyError("Actions not found after normalization.")
            norm_actions = np.asarray(normalized["actions"])
            if norm_actions.ndim == 1:
                norm_actions = norm_actions[None, :]

            post = _apply_transforms(
                normalized,
                [
                    *data_config.model_transforms.inputs,
                ],
            )
            if "actions" not in post:
                raise KeyError("Actions not found after model transforms.")
            post_actions = np.asarray(post["actions"])
            if post_actions.ndim == 1:
                post_actions = post_actions[None, :]

            pre_absmax = float(np.max(np.abs(pre_actions[..., args.dim])))
            norm_absmax = float(np.max(np.abs(norm_actions[..., args.dim])))
            post_absmax = float(np.max(np.abs(post_actions[..., args.dim])))

            task_index = _get_first_present(item, ["task_index", "task_id"])
            episode_index = _get_first_present(item, ["episode_index", "episode_id"])
            compare_results.append((post_absmax, norm_absmax, pre_absmax, idx, task_index, episode_index))
            if args.stop_threshold is not None and post_absmax > args.stop_threshold:
                print("Threshold exceeded in post-normalize scan:")
                print(
                    f"post_absmax={post_absmax:.6g} norm_absmax={norm_absmax:.6g} "
                    f"pre_absmax={pre_absmax:.6g} idx={idx}"
                    f"{'' if task_index is None else f' task_index={int(task_index)}'}"
                    f"{'' if episode_index is None else f' episode_index={int(episode_index)}'}"
                )
                return
            continue

        if args.action_key not in item:
            raise KeyError(f"Action key '{args.action_key}' not found in dataset item keys.")

        action = np.asarray(item[args.action_key])
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[-1] <= args.dim:
            raise ValueError(
                f"Action dim {args.dim} out of range for shape {action.shape}. "
                f"Check '--dim' or '--action-key'."
            )

        values = action[..., args.dim]
        absmax = float(np.max(np.abs(values)))

        # Optional metadata
        task_index = _get_first_present(item, ["task_index", "task_id"])
        episode_index = _get_first_present(item, ["episode_index", "episode_id"])

        results.append((absmax, idx, task_index, episode_index))
        if args.stop_threshold is not None and absmax > args.stop_threshold:
            print("Threshold exceeded in raw scan:")
            print(
                f"absmax={absmax:.6g} idx={idx}"
                f"{'' if task_index is None else f' task_index={int(task_index)}'}"
                f"{'' if episode_index is None else f' episode_index={int(episode_index)}'}"
            )
            return

    if args.compare_normalize:
        compare_results.sort(key=lambda x: x[0], reverse=True)
        print(f"Scanned {total} samples from {data_config.repo_id}")
        print(f"Compare normalize: actions dim {args.dim} (post vs norm vs pre)")
        print("Top outliers by post-model abs max:")
        for rank, (post_absmax, norm_absmax, pre_absmax, idx, task_index, episode_index) in enumerate(
            compare_results[:top_k], start=1
        ):
            print(
                f"{rank:02d} post_absmax={post_absmax:.6g} norm_absmax={norm_absmax:.6g} "
                f"pre_absmax={pre_absmax:.6g} idx={idx}"
                f"{'' if task_index is None else f' task_index={int(task_index)}'}"
                f"{'' if episode_index is None else f' episode_index={int(episode_index)}'}"
            )
        return

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"Scanned {total} samples from {data_config.repo_id}")
    print(f"Action key: {args.action_key}, dim: {args.dim}")
    print("Top outliers (abs max):")
    for rank, (value, idx, task_index, episode_index) in enumerate(results[:top_k], start=1):
        print(
            f"{rank:02d} absmax={value:.6g} idx={idx}"
            f"{'' if task_index is None else f' task_index={int(task_index)}'}"
            f"{'' if episode_index is None else f' episode_index={int(episode_index)}'}"
        )


if __name__ == "__main__":
    main()
