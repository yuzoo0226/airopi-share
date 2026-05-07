#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import openpi.models.model as _model
from openpi.policies import policy as policy_lib
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
import openpi.transforms as transforms
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as train_config
from openpi.training import experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve OpenPI policy as websocket server for HSR client")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config-name", default=None, help="Train config name registered in openpi.training.config")
    parser.add_argument(
        "--config-yaml",
        default=None,
        help="Experiment YAML path inside the container. When set, this takes precedence over --config-name.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--default-prompt", default=None, help="Fallback prompt if prompt key is missing")
    parser.add_argument("--record-dir", default=None, help="Optional directory for policy records")
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help='Optional torch device override (e.g. "cuda", "cuda:0", "cpu")',
    )
    return parser.parse_args()


def _autodetect_experiment_yaml(checkpoint_dir: str | None) -> Path | None:
    """Look for a checkpoint-embedded experiment_config.yaml.

    Training (`save_state`) writes it via orbax's `experiment_config` item to
    `<step_dir>/experiment_config/experiment_config.yaml`. As a courtesy also
    check `<step_dir>/experiment_config.yaml` for manually-placed files.
    """
    if not checkpoint_dir:
        return None
    base = Path(str(checkpoint_dir)).expanduser()
    candidates = [
        base / "experiment_config" / "experiment_config.yaml",
        base / "experiment_config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_train_config(args: argparse.Namespace) -> tuple[train_config.TrainConfig, str, str | None]:
    config_yaml = str(getattr(args, "config_yaml", None) or "").strip()
    config_name = str(getattr(args, "config_name", None) or "").strip()
    checkpoint_dir = str(getattr(args, "checkpoint_dir", None) or "").strip()

    if not config_yaml and not config_name:
        detected = _autodetect_experiment_yaml(checkpoint_dir)
        if detected is not None:
            logging.info("Auto-detected experiment config from checkpoint: %s", detected)
            config_yaml = str(detected)

    if config_yaml:
        yaml_path = Path(config_yaml).expanduser()
        if not yaml_path.is_absolute():
            yaml_path = (Path.cwd() / yaml_path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"config_yaml not found: {yaml_path}")
        config = experiment_config.ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()
        return config, config.name, str(yaml_path)

    if not config_name:
        raise ValueError(
            "Could not resolve training config. Pass --config-name or --config-yaml, "
            "or ensure the checkpoint embeds experiment_config/experiment_config.yaml "
            "(produced automatically by scripts/train.py with --config-yaml)."
        )
    return train_config.get_config(config_name), config_name, None


def _create_trained_policy(
    config: train_config.TrainConfig,
    checkpoint_dir: str | Path,
    *,
    default_prompt: str | None,
    pytorch_device: str | None,
) -> policy_lib.Policy:
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    weight_path = checkpoint_dir / "model.safetensors"
    is_pytorch = weight_path.exists()

    logging.info("Loading model...")
    if is_pytorch:
        model = config.model.load_pytorch(config, str(weight_path))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats_dir = checkpoint_dir / "assets" / data_config.asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info("Loaded norm stats from %s", norm_stats_dir)

    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return policy_lib.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def _make_policy(
    *,
    checkpoint_dir: str,
    config_name: str | None = None,
    config_yaml: str | None = None,
    pytorch_device: str | None = None,
    default_prompt: str | None = None,
    record_dir: str | None = None,
) -> tuple[policy_lib.Policy | policy_lib.PolicyRecorder, dict[str, Any] | None]:
    """Create a policy from checkpoint. Returns (policy, metadata_updates)."""
    # Build a minimal namespace for _resolve_train_config
    ns = argparse.Namespace(
        config_name=config_name,
        config_yaml=config_yaml,
        checkpoint_dir=checkpoint_dir,
    )
    config, resolved_name, resolved_yaml = _resolve_train_config(ns)

    policy = _create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )

    if record_dir:
        policy = policy_lib.PolicyRecorder(policy, record_dir)

    metadata_updates = {
        "config_name": resolved_name,
        "config_yaml": resolved_yaml or "",
        "checkpoint_dir": checkpoint_dir,
    }
    return policy, metadata_updates


def main() -> None:
    args = parse_args()

    # Enable persistent JAX compilation cache so JIT-compiled kernels survive
    # container restarts and hot-reloads with the same model architecture.
    jax_cache_dir = str(Path("~/.cache/jax").expanduser())
    os.makedirs(jax_cache_dir, exist_ok=True)
    try:
        jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
    except Exception:
        # Cache corruption — clear and retry
        import shutil
        shutil.rmtree(jax_cache_dir, ignore_errors=True)
        os.makedirs(jax_cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", jax_cache_dir)

    checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    config, resolved_config_name, resolved_yaml_path = _resolve_train_config(args)

    policy = _create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )

    if args.record_dir:
        policy = policy_lib.PolicyRecorder(policy, args.record_dir)

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "config_name": resolved_config_name,
            "requested_config_name": str(args.config_name or ""),
            "config_yaml": resolved_yaml_path or "",
            "checkpoint_dir": checkpoint_dir,
            "server_host": args.host,
            "server_port": args.port,
        }
    )

    def policy_factory(
        *,
        checkpoint_dir: str,
        config_name: str | None = None,
        config_yaml: str | None = None,
        pytorch_device: str | None = None,
        default_prompt: str | None = None,
    ) -> tuple:
        return _make_policy(
            checkpoint_dir=checkpoint_dir,
            config_name=config_name,
            config_yaml=config_yaml,
            pytorch_device=pytorch_device,
            default_prompt=default_prompt or args.default_prompt,
            record_dir=args.record_dir,
        )

    logging.info(
        "Serving policy config=%s yaml=%s checkpoint=%s on %s:%s",
        resolved_config_name,
        resolved_yaml_path or "<none>",
        checkpoint_dir,
        args.host,
        args.port,
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        policy_factory=policy_factory,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
