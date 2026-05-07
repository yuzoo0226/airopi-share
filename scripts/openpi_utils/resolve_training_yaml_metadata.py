#!/usr/bin/env python3
"""Resolve experiment YAML metadata used by launcher scripts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import pathlib
from typing import Any, Optional, Set

import yaml


OUTPUT_KEYS = (
    "DATASET_DATA_DIR",
    "DATASET_REPO_ID",
    "DATASET_ASSETS_DIR",
    "DATASET_ASSET_ID",
    "GPU_NUM_GPUS",
    "BASE_MODEL_URL",
    "DATASET_HF_HOME",
)


def deep_merge(base: Any, override: Any) -> Any:
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        return override

    merged = dict(base)
    for key, value in override.items():
        if key in merged:
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_inheritance(yaml_path: pathlib.Path, seen: Optional[Set[pathlib.Path]] = None) -> dict[str, Any]:
    yaml_path = yaml_path.expanduser().resolve()
    seen = set() if seen is None else seen
    if yaml_path in seen:
        raise RuntimeError(f"Cyclic _base_ reference detected: {yaml_path}")
    seen.add(yaml_path)

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML document must be a mapping: {yaml_path}")

    base_ref = data.pop("_base_", None)
    if not base_ref:
        return data

    base_path = pathlib.Path(base_ref)
    if not base_path.is_absolute():
        base_path = (yaml_path.parent / base_path).resolve()
    base_data = load_yaml_with_inheritance(base_path, seen)
    return deep_merge(base_data, data)


def derive_dataset_hf_home(data_dir: str, repo_id: str) -> str:
    if not data_dir or not repo_id:
        return ""

    data_path = pathlib.Path(data_dir).expanduser().resolve()
    repo_parts = [part for part in repo_id.split("/") if part]
    if not repo_parts:
        return ""

    repo_tail = pathlib.Path(*repo_parts)
    if not data_path.as_posix().endswith(repo_tail.as_posix()):
        return ""

    hf_root = data_path
    for _ in repo_parts:
        hf_root = hf_root.parent
    return str(hf_root)


def extract_metadata(config: Mapping[str, Any]) -> dict[str, str]:
    dataset = config.get("dataset") or {}
    gpu = config.get("gpu") or {}
    checkpoints = (config.get("checkpoints") or {}).get("base_model") or {}
    model_type = (config.get("model") or {}).get("type") or ""

    data_dir = str(dataset.get("data_dir") or "")
    repo_id = str(dataset.get("repo_id") or "")
    return {
        "DATASET_DATA_DIR": data_dir,
        "DATASET_REPO_ID": repo_id,
        "DATASET_ASSETS_DIR": str(dataset.get("assets_dir") or ""),
        "DATASET_ASSET_ID": str(dataset.get("asset_id") or ""),
        "GPU_NUM_GPUS": str(gpu.get("num_gpus") or ""),
        "BASE_MODEL_URL": str(checkpoints.get(model_type, "") or ""),
        "DATASET_HF_HOME": derive_dataset_hf_home(data_dir, repo_id),
    }


def format_metadata_lines(metadata: Mapping[str, str]) -> str:
    return "\n".join(f"{key}={metadata.get(key, '')}" for key in OUTPUT_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve launcher metadata from an experiment YAML file.")
    parser.add_argument("config_yaml", help="Path to the experiment YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_with_inheritance(pathlib.Path(args.config_yaml))
    print(format_metadata_lines(extract_metadata(config)))


if __name__ == "__main__":
    main()
