#!/usr/bin/env python3
"""Evaluate validation loss and action MSE on a val dataset."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _resolve_checkpoint_step(checkpoint_manager: Any, step: int | None) -> int:
    if step is not None:
        return step
    steps = tuple(checkpoint_manager.all_steps())
    if not steps:
        raise FileNotFoundError("No checkpoints found in checkpoint directory.")
    return max(steps)


def _resolve_checkpoint_paths(checkpoint_root: epath.Path, step: int | None) -> tuple[epath.Path, int]:
    checkpoint_root = epath.Path(checkpoint_root)
    if (checkpoint_root / "params").exists():
        # If user passed a step directory, jump to its parent.
        checkpoint_root = checkpoint_root.parent
    if checkpoint_root.name.isdigit():
        checkpoint_root = checkpoint_root.parent

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        checkpoint_root,
        keep_period=None,
        overwrite=False,
        resume=True,
    )
    resolved_step = _resolve_checkpoint_step(checkpoint_manager, step)
    return checkpoint_root / str(resolved_step) / "params", resolved_step


def _override_data_config(
    config: _config.TrainConfig,
    *,
    val_repo_id: str | None,
    asset_id: str | None,
    asset_dir: str | None,
) -> _config.TrainConfig:
    if val_repo_id is None:
        return config

    data_factory = config.data
    assets = data_factory.assets
    if asset_id is None:
        asset_id = assets.asset_id or data_factory.repo_id
    assets = dataclasses.replace(assets, asset_id=asset_id, assets_dir=asset_dir or assets.assets_dir)
    data_factory = dataclasses.replace(data_factory, repo_id=val_repo_id, assets=assets)
    return dataclasses.replace(config, data=data_factory)


def _pad_to_dim(x: jax.Array, target_dim: int, value: float) -> jax.Array:
    if x.shape[-1] == target_dim:
        return x
    if x.shape[-1] > target_dim:
        return x[..., :target_dim]
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, target_dim - x.shape[-1])]
    return jnp.pad(x, pad_width, constant_values=value)


def _make_eval_step(
    *,
    mean: jax.Array | None,
    std: jax.Array | None,
    q01: jax.Array | None,
    q99: jax.Array | None,
    use_quantiles: bool,
    num_sample_steps: int,
    model_def: nnx.GraphDef[_model.BaseModel],
):
    has_stats = mean is not None and std is not None

    def _unnormalize(actions: jax.Array) -> jax.Array:
        if not has_stats:
            return actions
        if use_quantiles:
            if q01 is None or q99 is None:
                return actions
            q01_p = q01
            q99_p = q99
            dim = q01_p.shape[-1]
            if dim < actions.shape[-1]:
                left = (actions[..., :dim] + 1.0) / 2.0 * (q99_p - q01_p + 1e-6) + q01_p
                return jnp.concatenate([left, actions[..., dim:]], axis=-1)
            return (actions + 1.0) / 2.0 * (q99_p - q01_p + 1e-6) + q01_p

        mean_p = _pad_to_dim(mean, actions.shape[-1], 0.0)
        std_p = _pad_to_dim(std, actions.shape[-1], 1.0)
        return actions * (std_p + 1e-6) + mean_p

    @at.typecheck
    def _eval_step(
        rng: at.KeyArrayLike,
        params_state: nnx.State,
        batch: tuple[_model.Observation, _model.Actions],
    ) -> dict[str, at.Array]:
        model = nnx.merge(model_def, params_state)
        model.eval()

        observation, actions = batch
        loss_rng, sample_rng = jax.random.split(rng)
        loss = jnp.mean(model.compute_loss(loss_rng, observation, actions, train=False))
        pred_actions = model.sample_actions(sample_rng, observation, num_steps=num_sample_steps)
        pred_actions = _unnormalize(pred_actions)
        actions = _unnormalize(actions)
        action_mse = jnp.mean(jnp.square(pred_actions - actions))
        return {"val_loss": loss, "action_mse": action_mse}

    return _eval_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation loss and action MSE.")
    parser.add_argument("config_name")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--val-repo-id", default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--asset-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-base-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--num-sample-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to evaluate (default: latest)")
    args = parser.parse_args()

    init_logging()
    logging.info(f"Running on: {platform.node()}")

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        wandb_enabled=False,
        assets_base_dir=args.assets_base_dir or config.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir or config.checkpoint_base_dir,
        batch_size=args.batch_size or config.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else config.num_workers,
        seed=args.seed,
    )
    config = _override_data_config(
        config,
        val_repo_id=args.val_repo_id,
        asset_id=args.asset_id,
        asset_dir=args.asset_dir,
    )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=args.num_batches,
    )
    data_config = data_loader.data_config()
    action_stats = None
    if data_config.norm_stats is not None:
        action_stats = data_config.norm_stats.get("actions")
    use_quantiles = bool(data_config.use_quantile_norm)
    mean = jnp.asarray(action_stats.mean) if action_stats is not None else None
    std = jnp.asarray(action_stats.std) if action_stats is not None else None
    q01 = jnp.asarray(action_stats.q01) if action_stats is not None and action_stats.q01 is not None else None
    q99 = jnp.asarray(action_stats.q99) if action_stats is not None and action_stats.q99 is not None else None

    checkpoint_root = epath.Path(args.checkpoint_dir or config.checkpoint_dir)
    params_dir, step = _resolve_checkpoint_paths(checkpoint_root, args.step)
    restored_params = _model.restore_params(params_dir, sharding=replicated_sharding)
    model = config.model.create(init_rng)
    model_def, params_state = nnx.split(model)
    params_state.replace_by_pure_dict(restored_params)

    eval_step = _make_eval_step(
        mean=mean,
        std=std,
        q01=q01,
        q99=q99,
        use_quantiles=use_quantiles,
        num_sample_steps=args.num_sample_steps,
        model_def=model_def,
    )
    peval_step = jax.jit(
        eval_step,
        in_shardings=(replicated_sharding, replicated_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    metrics = []
    for batch in data_loader:
        if isinstance(batch, tuple) and len(batch) == 3:
            batch = (batch[0], batch[1])
        rng, step_rng = jax.random.split(rng)
        with sharding.set_mesh(mesh):
            out = peval_step(step_rng, params_state, batch)
        metrics.append(out)

    if not metrics:
        raise ValueError("No batches were evaluated. Check --num-batches and dataset size.")

    stacked = common_utils.stack_forest(metrics)
    reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
    logging.info(
        "Eval results: "
        + ", ".join(f"{k}={v:.6f}" for k, v in reduced.items())
    )


if __name__ == "__main__":
    main()
