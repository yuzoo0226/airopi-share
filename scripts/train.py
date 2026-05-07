import argparse
import dataclasses
import functools
import json
import logging
import os
import platform
import pathlib
import random
import sys
import zlib
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.distributed as _distributed
from openpi.training import experiment_config as _experiment_config
from openpi.training.experiment_config import ExperimentConfig
from openpi.training.fast_lerobot_metadata import FastLeRobotDatasetMetadata
import openpi.training.loss_utils as loss_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.task_log_utils as task_log_utils
import openpi.training.task_sampler as _task_sampler
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


_HSR_ACTION_NAMES_11 = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "hand_motor_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "base_x",
    "base_y",
    "base_t",
]

_HSR_ACTION_NAMES_16 = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "unused_5",
    "hand_motor_joint",
    "unused_7",
    "unused_8",
    "unused_9",
    "unused_10",
    "head_pan_joint",
    "head_tilt_joint",
    "base_x",
    "base_y",
    "base_t",
]


def init_logging():
    """Custom logging format for better readability."""
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


def _parse_visible_cuda_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [token.strip() for token in raw.split(",") if token.strip()]


def _log_device_diagnostics(dist_ctx: _distributed.DistributedContext) -> tuple[int, int]:
    """Log per-rank device visibility to pinpoint missing GPU/device issues."""
    local_devices = list(jax.local_devices())
    visible_devices = _parse_visible_cuda_devices()

    local_device_ids: list[int | str] = []
    for device in local_devices:
        dev_id = getattr(device, "id", None)
        local_device_ids.append(dev_id if dev_id is not None else repr(device))

    missing_visible_ids: list[str] = []
    try:
        visible_int = {int(v) for v in visible_devices}
        local_int = {int(v) for v in local_device_ids if isinstance(v, int)}
        missing_visible_ids = [str(v) for v in sorted(visible_int - local_int)]
    except ValueError:
        # CUDA_VISIBLE_DEVICES can be UUID-form on some schedulers.
        pass

    global_device_count = jax.device_count()
    local_device_count = jax.local_device_count()

    logging.info(
        "Device diagnostics: host=%s slurmd_nodename=%s rank=%d/%d local_rank=%d local_world_size=%d "
        "CUDA_VISIBLE_DEVICES=%s local_device_ids=%s local_device_count=%d global_device_count=%d "
        "missing_visible_device_ids=%s",
        platform.node(),
        os.environ.get("SLURMD_NODENAME", ""),
        dist_ctx.rank,
        dist_ctx.world_size,
        dist_ctx.local_rank,
        dist_ctx.local_world_size,
        ",".join(visible_devices) if visible_devices else "<unset>",
        local_device_ids,
        local_device_count,
        global_device_count,
        ",".join(missing_visible_ids) if missing_visible_ids else "<none>",
    )

    return local_device_count, global_device_count


def _pad_to_dim(x: jnp.ndarray, target_dim: int, value: float) -> jnp.ndarray:
    if x.shape[-1] == target_dim:
        return x
    if x.shape[-1] > target_dim:
        return x[..., :target_dim]
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, target_dim - x.shape[-1])]
    return jnp.pad(x, pad_width, constant_values=value)


def _stable_task_seed(task: str, seed: int) -> int:
    return (seed + zlib.crc32(task.encode("utf-8"))) & 0xFFFF_FFFF


def _episode_task_label(meta: LeRobotDatasetMetadata, episode: dict) -> str:
    tasks = episode.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if isinstance(tasks, str) and tasks:
        return tasks
    if "task" in episode and episode["task"]:
        return str(episode["task"])
    task_index = episode.get("task_index")
    if isinstance(task_index, int) and isinstance(meta.tasks, dict) and task_index in meta.tasks:
        return str(meta.tasks[task_index])
    if task_index is not None:
        return str(task_index)
    return "unknown"


def _split_episodes_by_task(
    repo_id: str,
    val_fraction: float,
    seed: int,
    *,
    dataset_root: str | None = None,
) -> tuple[list[int], list[int]]:
    split_cache_path = _episode_split_cache_path(repo_id, val_fraction, seed)
    cached_split = _load_episode_split_cache(split_cache_path, repo_id=repo_id)
    if cached_split is not None:
        train_eps, val_eps = cached_split
        logging.info(
            "Loaded episode split cache: train=%d val=%d path=%s",
            len(train_eps),
            len(val_eps),
            split_cache_path,
        )
        return train_eps, val_eps

    meta = None
    if dataset_root:
        local_root = pathlib.Path(dataset_root).resolve()
        try:
            meta = FastLeRobotDatasetMetadata(repo_id, root=local_root)
            logging.info("Loaded LeRobot metadata from local root: %s", local_root)
        except (FileNotFoundError, NotADirectoryError) as exc:
            logging.warning("Failed to load local dataset metadata from %s: %s", local_root, exc)

    if meta is None:
        meta = FastLeRobotDatasetMetadata(repo_id)
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx, episode in meta.episodes.items():
        label = _episode_task_label(meta, episode)
        task_to_episodes.setdefault(label, []).append(int(ep_idx))

    train_eps: list[int] = []
    val_eps: list[int] = []
    for task, episodes in sorted(task_to_episodes.items()):
        rng = random.Random(_stable_task_seed(task, seed))
        eps = episodes[:]
        rng.shuffle(eps)
        if len(eps) < 2:
            val_count = 0
        else:
            # Keep at least one episode in both train and val when possible.
            val_count = int(len(eps) * val_fraction)
            val_count = max(1, min(val_count, len(eps) - 1))
        val_eps.extend(eps[:val_count])
        train_eps.extend(eps[val_count:])

    _save_episode_split_cache(split_cache_path, repo_id=repo_id, train_eps=train_eps, val_eps=val_eps)

    return train_eps, val_eps


def _dataset_episodes_fingerprint(repo_id: str) -> str | None:
    base_dir = os.environ.get("HF_LEROBOT_HOME")
    if not base_dir:
        return None
    episodes_meta = pathlib.Path(base_dir) / repo_id / "meta" / "episodes.jsonl"
    if not episodes_meta.exists():
        return None
    stat = episodes_meta.stat()
    return f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}"


def _episode_split_cache_path(repo_id: str, val_fraction: float, seed: int) -> pathlib.Path:
    cache_dir = pathlib.Path(os.environ.get("OPENPI_EPISODE_SPLIT_CACHE_DIR", "tmp_storage/episode_split_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{repo_id}|{val_fraction:.12f}|{seed}"
    key_hash = zlib.crc32(key.encode("utf-8")) & 0xFFFF_FFFF
    return cache_dir / f"split_{key_hash:08x}.json"


def _load_episode_split_cache(cache_path: pathlib.Path, *, repo_id: str) -> tuple[list[int], list[int]] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        saved_repo_id = payload.get("repo_id")
        if saved_repo_id != repo_id:
            return None

        expected_fp = _dataset_episodes_fingerprint(repo_id)
        saved_fp = payload.get("dataset_fingerprint")
        if expected_fp is not None and saved_fp != expected_fp:
            logging.warning(
                "Episode split cache fingerprint mismatch. expected=%s saved=%s path=%s",
                expected_fp,
                saved_fp,
                cache_path,
            )
            return None
        train_eps = [int(x) for x in payload["train_eps"]]
        val_eps = [int(x) for x in payload["val_eps"]]
        return train_eps, val_eps
    except Exception as exc:
        logging.warning("Failed to load episode split cache from %s: %s", cache_path, exc)
        return None


def _save_episode_split_cache(
    cache_path: pathlib.Path,
    *,
    repo_id: str,
    train_eps: list[int],
    val_eps: list[int],
) -> None:
    payload = {
        "repo_id": repo_id,
        "dataset_fingerprint": _dataset_episodes_fingerprint(repo_id),
        "train_eps": train_eps,
        "val_eps": val_eps,
    }
    try:
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(cache_path)
    except Exception as exc:
        logging.warning("Failed to save episode split cache to %s: %s", cache_path, exc)


def _make_eval_step(
    *,
    mean: jnp.ndarray | None,
    std: jnp.ndarray | None,
    q01: jnp.ndarray | None,
    q99: jnp.ndarray | None,
    use_quantiles: bool,
    num_sample_steps: int,
):
    has_stats = mean is not None and std is not None

    def _unnormalize(actions: jnp.ndarray) -> jnp.ndarray:
        if not has_stats:
            return actions
        if use_quantiles:
            if q01 is None or q99 is None:
                return actions
            dim = q01.shape[-1]
            if dim < actions.shape[-1]:
                left = (actions[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
                return jnp.concatenate([left, actions[..., dim:]], axis=-1)
            return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

        mean_p = _pad_to_dim(mean, actions.shape[-1], 0.0)
        std_p = _pad_to_dim(std, actions.shape[-1], 1.0)
        return actions * (std_p + 1e-6) + mean_p

    @at.typecheck
    def _eval_step(
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[_model.Observation, _model.Actions],
    ) -> dict[str, at.Array]:
        params_state = state.ema_params if state.ema_params is not None else state.params
        model = nnx.merge(state.model_def, params_state)
        model.eval()

        observation, actions = batch
        loss_rng, sample_rng = jax.random.split(rng)
        loss = jnp.mean(model.compute_loss(loss_rng, observation, actions, train=False))
        pred_actions = model.sample_actions(sample_rng, observation, num_steps=num_sample_steps)
        pred_actions = _unnormalize(pred_actions)
        actions = _unnormalize(actions)
        abs_error = jnp.abs(pred_actions - actions)
        action_l1 = jnp.mean(abs_error)
        action_mse = jnp.mean(jnp.square(pred_actions - actions))
        reduce_axes = tuple(range(abs_error.ndim - 1))
        action_l1_per_dim = jnp.mean(abs_error, axis=reduce_axes)
        return {
            "val_loss": loss,
            "action_l1": action_l1,
            "action_l1_per_dim": action_l1_per_dim,
            "action_mse": action_mse,
        }

    return _eval_step


def _metric_safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def _resolve_action_dim_labels(config: _config.TrainConfig, dim_count: int) -> list[str]:
    labels: list[str] = []
    if isinstance(config.data, _config.LeRobotHSRDataConfig):
        if config.data.adapt_to_pi and dim_count >= 16:
            labels.extend(_HSR_ACTION_NAMES_16)
        elif dim_count >= 11:
            labels.extend(_HSR_ACTION_NAMES_11)

    labels = labels[:dim_count]
    if len(labels) < dim_count:
        labels.extend([f"dim_{i}" for i in range(len(labels), dim_count)])
    return labels


def _format_eval_log_values(
    reduced_vals: dict[str, Any],
    *,
    action_dim_labels: list[str],
) -> tuple[dict[str, float], str]:
    """Flatten eval metrics for wandb logging with per-dimension action L1 values."""
    log_vals: dict[str, float] = {}
    for key, value in reduced_vals.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            log_vals[f"val/{key}"] = float(arr)
            continue
        if key == "action_l1_per_dim":
            flat = arr.reshape(-1)
            for i, v in enumerate(flat):
                dim_label = action_dim_labels[i] if i < len(action_dim_labels) else f"dim_{i}"
                log_vals[f"val/action_l1/{_metric_safe_name(dim_label)}"] = float(v)

    summary_order = ["val/val_loss", "val/action_l1", "val/action_mse"]
    summary_parts = [f"{k}={log_vals[k]:.4f}" for k in summary_order if k in log_vals]
    for i in range(3):
        dim_label = action_dim_labels[i] if i < len(action_dim_labels) else f"dim_{i}"
        k = f"val/action_l1/{_metric_safe_name(dim_label)}"
        if k in log_vals:
            summary_parts.append(f"{k}={log_vals[k]:.4f}")
    return log_vals, ", ".join(summary_parts)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    experiment_yaml: str | None = None,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

    # Attach the merged experiment YAML to the wandb run files so collaborators
    # can copy the exact config without pulling the checkpoint.
    if experiment_yaml is not None:
        yaml_path = ckpt_dir / "experiment_config.yaml"
        yaml_path.write_text(experiment_yaml)
        wandb.save(str(yaml_path), base_path=str(ckpt_dir), policy="now")


def load_train_config_from_cli() -> tuple[_config.TrainConfig, str | None]:
    """Load TrainConfig from either tyro config-name CLI or YAML.

    Returns (TrainConfig, merged_yaml_text). The yaml text is None when the
    tyro config-name path is used; otherwise it is the `_base_`-resolved,
    single-file YAML string to embed in each saved checkpoint.
    """
    has_yaml_flag = any(arg == "--config-yaml" or arg.startswith("--config-yaml=") for arg in sys.argv[1:])
    if not has_yaml_flag:
        return _config.cli(), None

    parser = argparse.ArgumentParser(description="Unified training entrypoint with YAML support.")
    parser.add_argument("--config-yaml", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--exp-name", default=None, help="Override experiment name.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoint directory.")
    parser.add_argument("--checkpoint-base-dir", default=None, help="Override checkpoint base directory.")
    parser.add_argument("--assets-base-dir", default=None, help="Override assets base directory.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")

    args, unknown = parser.parse_known_args(sys.argv[1:])
    if unknown:
        raise ValueError(f"Unknown CLI args in YAML mode: {' '.join(unknown)}")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")

    # In multi-node runs, JAX distributed must be initialized before backend-touching
    # calls (e.g. jax.device_count()) inside GPU auto-detection.
    exp_config = ExperimentConfig.from_yaml(args.config_yaml, auto_detect_gpu=False)
    if exp_config.dataset.data_dir:
        os.environ["OPENPI_DATASET_ROOT"] = str(pathlib.Path(exp_config.dataset.data_dir).resolve())
    if not os.environ.get("HF_LEROBOT_HOME") and exp_config.dataset.data_dir and exp_config.dataset.repo_id:
        dataset_dir = pathlib.Path(exp_config.dataset.data_dir).resolve()
        repo_parts = [part for part in exp_config.dataset.repo_id.split("/") if part]
        repo_tail = pathlib.Path(*repo_parts) if repo_parts else None
        if repo_tail is not None and dataset_dir.as_posix().endswith(repo_tail.as_posix()):
            hf_home = dataset_dir
            for _ in repo_parts:
                hf_home = hf_home.parent
            os.environ["HF_LEROBOT_HOME"] = str(hf_home)
            logging.info("HF_LEROBOT_HOME was set from YAML dataset.data_dir: %s", hf_home)

    config = exp_config.to_train_config()
    replacements: dict[str, object] = {}
    if args.exp_name is not None:
        replacements["exp_name"] = args.exp_name
    if args.resume:
        replacements["resume"] = True
    if args.overwrite:
        replacements["overwrite"] = True
    if args.checkpoint_base_dir is not None:
        replacements["checkpoint_base_dir"] = args.checkpoint_base_dir
    if args.assets_base_dir is not None:
        replacements["assets_base_dir"] = args.assets_base_dir
    if args.seed is not None:
        replacements["seed"] = args.seed
    if replacements:
        config = dataclasses.replace(config, **replacements)
    try:
        yaml_text = _experiment_config.dump_merged_yaml(args.config_yaml)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to serialize merged experiment YAML (%s); checkpoint will not embed it.", exc)
        yaml_text = None
    return config, yaml_text


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _next_batch(
    data_iter,
    *,
    data_loader=None,
    allow_errors: bool,
    max_errors: int,
    error_state: dict[str, int | str | None],
):
    """Fetch the next batch, optionally skipping errors from missing/corrupt data.

    Returns (batch, maybe-reset iterator) so callers can keep iterating.
    """
    def _format_error(exc: BaseException) -> str:
        message = str(exc).strip()
        return f"{type(exc).__name__}: {message}" if message else type(exc).__name__

    def _describe_loader(loader) -> str:
        if loader is None:
            return "None"
        name = type(loader).__name__
        inner = getattr(loader, "_data_loader", None)
        if inner is not None:
            name = f"{name}<{type(inner).__name__}>"
        return name

    while True:
        try:
            return next(data_iter), data_iter
        except _task_sampler.TaskSamplerInvariantError:
            # Invariant violations are deterministic and should never be retried.
            raise
        except StopIteration:
            if data_loader is None:
                raise
            error_state["count"] += 1
            last_error = "StopIteration (iterator exhausted)"
            error_state["last_error"] = last_error
            logging.warning(
                "DataLoader iterator exhausted or invalidated (error %d/%d); recreating iterator. loader=%s; "
                "last_error=%s",
                error_state["count"],
                max_errors,
                _describe_loader(data_loader),
                last_error,
            )
            if error_state["count"] >= max_errors:
                raise RuntimeError(
                    f"Too many data errors while fetching batches. Last error: {last_error}"
                ) from None
            data_iter = iter(data_loader)
        except Exception as exc:  # noqa: BLE001
            if not allow_errors:
                raise
            error_state["count"] += 1
            last_error = _format_error(exc)
            error_state["last_error"] = last_error
            logging.warning(
                "Skipping bad batch (error %d/%d). loader=%s; last_error=%s",
                error_state["count"],
                max_errors,
                _describe_loader(data_loader),
                last_error,
            )
            if error_state["count"] >= max_errors:
                raise RuntimeError(
                    f"Too many data errors while fetching batches. Last error: {last_error}"
                ) from exc
            if data_loader is not None:
                data_iter = iter(data_loader)


def _to_process_local_numpy(value) -> np.ndarray:
    """Convert possibly global jax.Array into process-local host ndarray."""
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        shards = sorted(
            value.addressable_shards,
            key=lambda shard: tuple(
                idx if isinstance(idx, int) else (0 if idx.start is None else idx.start)
                for idx in shard.index
            ),
        )
        if not shards:
            return np.asarray([])
        return np.concatenate([np.asarray(shard.data) for shard in shards], axis=0)
    return np.asarray(jax.device_get(value))


def _task_sampler_supports_update(data_loader) -> bool:
    task_sampler = getattr(data_loader, "_task_sampler", None)
    return task_sampler is not None and hasattr(task_sampler, "update")


def _adaptive_task_sampler(data_loader) -> _task_sampler.AdaptiveTaskSampler | None:
    task_sampler = getattr(data_loader, "_task_sampler", None)
    if isinstance(task_sampler, _task_sampler.AdaptiveTaskSampler):
        return task_sampler
    return None


def _sync_task_feedback(task_indices_local: np.ndarray, losses_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    task_indices_local = np.asarray(task_indices_local, dtype=np.int64).reshape(-1)
    losses_local = np.asarray(losses_local, dtype=np.float64).reshape(-1)
    if jax.process_count() <= 1:
        return task_indices_local, losses_local
    task_indices = _distributed.allgather_processes_1d(task_indices_local)
    losses = _distributed.allgather_processes_1d(losses_local)
    return task_indices, losses


def _concat_feedback_chunks(chunks: list[np.ndarray], *, dtype: np.dtype) -> np.ndarray:
    if not chunks:
        return np.asarray([], dtype=dtype)
    if len(chunks) == 1:
        return np.asarray(chunks[0], dtype=dtype).reshape(-1)
    return np.concatenate([np.asarray(chunk, dtype=dtype).reshape(-1) for chunk in chunks], axis=0)


def _assert_task_feedback_compatible(task_indices: np.ndarray, losses: np.ndarray, *, rank: int) -> None:
    if task_indices.shape[0] != losses.shape[0]:
        raise RuntimeError(
            "Per-task feedback shape mismatch: "
            f"task_indices={task_indices.shape[0]}, losses={losses.shape[0]}, rank={rank}."
        )


def _accumulate_task_loss_stats(
    task_indices: np.ndarray,
    losses: np.ndarray,
    task_loss_sums: dict[int, float],
    task_loss_counts: dict[int, int],
) -> None:
    task_indices = np.asarray(task_indices, dtype=np.int64).reshape(-1)
    losses = np.asarray(losses, dtype=np.float64).reshape(-1)
    if task_indices.size == 0:
        return
    for task_id in np.unique(task_indices):
        mask = task_indices == task_id
        task_loss_sums[int(task_id)] = task_loss_sums.get(int(task_id), 0.0) + float(losses[mask].sum())
        task_loss_counts[int(task_id)] = task_loss_counts.get(int(task_id), 0) + int(mask.sum())


def _sync_adaptive_task_sampler_state(data_loader) -> bool:
    task_sampler = _adaptive_task_sampler(data_loader)
    if task_sampler is None:
        return False
    loss_ema = task_sampler.get_loss_ema()
    synced_loss_ema = _distributed.allreduce_mean_processes_1d(loss_ema)
    task_sampler.set_loss_ema(synced_loss_ema)
    return True


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step_with_per_example_loss(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array], at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        per_example_loss = loss_utils.reduce_per_example_loss(chunked_loss)
        return jnp.mean(per_example_loss), per_example_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, per_example_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info, per_example_loss


def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Backward-compatible wrapper for callers that expect only state and metrics."""
    new_state, info, _ = train_step_with_per_example_loss(config, rng, state, batch)
    return new_state, info


def main(config: _config.TrainConfig, experiment_yaml: str | None = None):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    allow_data_errors = os.environ.get("OPENPI_SKIP_DATA_ERRORS", "1") == "1"
    max_data_errors = int(os.environ.get("OPENPI_MAX_DATA_ERRORS", "100"))

    dist_ctx = _distributed.initialize_jax_distributed_from_env()
    global_task_rank = dist_ctx.rank
    n_tasks = dist_ctx.world_size
    _, global_device_count = _log_device_diagnostics(dist_ctx)

    if config.batch_size % global_device_count != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {global_device_count}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_dir_path = epath.Path(config.checkpoint_dir).resolve()
    checkpoint_dir_has_contents = checkpoint_dir_path.exists() and any(checkpoint_dir_path.iterdir())
    if dist_ctx.is_primary and checkpoint_dir_has_contents and config.overwrite and n_tasks > 1:
        logging.info("Primary rank will overwrite existing checkpoint directory for distributed run: %s", checkpoint_dir_path)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(
        config,
        resuming=resuming,
        enabled=(config.wandb_enabled and dist_ctx.is_primary),
        experiment_yaml=experiment_yaml,
    )
    if dist_ctx.is_primary:
        logging.info("Initialized wandb.")
    else:
        logging.info("Initialized wandb in disabled mode for non-primary rank %d.", dist_ctx.rank)

    data_config = config.data.create(config.assets_dirs, config.model)
    train_episodes: list[int] | None = None
    val_episodes: list[int] | None = None
    if config.val_split_fraction is not None:
        if not (0.0 < config.val_split_fraction < 1.0):
            raise ValueError("val_split_fraction must be between 0 and 1.")
        if data_config.repo_id is None or data_config.repo_id == "fake" or data_config.rlds_data_dir is not None:
            logging.warning("Validation split requested, but dataset does not support episode filtering.")
        else:
            if dist_ctx.is_primary:
                train_episodes, val_episodes = _split_episodes_by_task(
                    data_config.repo_id,
                    config.val_split_fraction,
                    config.val_split_seed,
                    dataset_root=os.environ.get("OPENPI_DATASET_ROOT"),
                )
                logging.info(
                    "Episode split: train=%d val=%d",
                    len(train_episodes),
                    len(val_episodes),
                )
            train_episodes, val_episodes = _distributed.broadcast_episode_split(
                train_episodes,
                val_episodes,
                is_source=dist_ctx.is_primary,
            )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        episodes=train_episodes,
        include_task_index=True,
    )
    data_iter = iter(data_loader)
    error_state = {"count": 0, "last_error": None}
    batch, data_iter = _next_batch(
        data_iter,
        data_loader=data_loader,
        allow_errors=allow_data_errors,
        max_errors=max_data_errors,
        error_state=error_state,
    )
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    if dist_ctx.is_primary and n_tasks == 1:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)
    elif dist_ctx.is_primary:
        logging.info("Skipping startup camera view logging for multi-node run.")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step_with_per_example_loss, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding, data_sharding),
        donate_argnums=(1,),
    )

    val_data_loader = None
    peval_step = None
    if config.eval_interval is not None and val_episodes:
        val_data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=config.eval_num_batches,
            episodes=val_episodes,
            enable_task_sampler=False,
        )
        action_stats = None
        if data_config.norm_stats is not None:
            action_stats = data_config.norm_stats.get("actions")
        mean = jnp.asarray(action_stats.mean) if action_stats is not None else None
        std = jnp.asarray(action_stats.std) if action_stats is not None else None
        q01 = jnp.asarray(action_stats.q01) if action_stats is not None and action_stats.q01 is not None else None
        q99 = jnp.asarray(action_stats.q99) if action_stats is not None and action_stats.q99 is not None else None
        eval_step = _make_eval_step(
            mean=mean,
            std=std,
            q01=q01,
            q99=q99,
            use_quantiles=data_config.use_quantile_norm,
            num_sample_steps=config.eval_num_sample_steps,
        )
        peval_step = jax.jit(
            eval_step,
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=not dist_ctx.is_primary,
    )

    infos = []
    last_task_distribution = None
    task_count_sum: int | None = None
    task_loss_sums: dict[int, float] = {}
    task_loss_counts: dict[int, int] = {}
    task_feedback_mode = "local_per_step"
    adaptive_feedback_mode = "global_per_step"
    adaptive_sync_interval = int(config.task_sampler.adaptive_sync_interval)
    if config.task_sampler.kind in ("none", "uniform") and jax.process_count() > 1:
        task_feedback_mode = "global_per_log_interval"
    if (
        config.task_sampler.kind == "adaptive"
        and config.task_sampler.adaptive_sync_mode == "local_periodic"
        and jax.process_count() > 1
    ):
        if adaptive_sync_interval <= 0:
            raise ValueError(
                "task_sampler.adaptive_sync_interval must be positive when "
                "adaptive_sync_mode=local_periodic, "
                f"got {config.task_sampler.adaptive_sync_interval}."
            )
        if _adaptive_task_sampler(data_loader) is None:
            logging.warning(
                "adaptive_sync_mode=local_periodic requested, but adaptive task sampler is unavailable. "
                "Falling back to global_per_step."
            )
        else:
            adaptive_feedback_mode = "local_periodic"
    if config.task_sampler.kind == "adaptive":
        task_feedback_mode = "global_per_step" if adaptive_feedback_mode == "global_per_step" else "local_per_step"
    if config.task_sampler.kind == "adaptive":
        logging.info(
            "Adaptive task sampler sync mode=%s (feedback_mode=%s, interval=%d, process_count=%d).",
            adaptive_feedback_mode,
            task_feedback_mode,
            adaptive_sync_interval,
            jax.process_count(),
        )
    elif config.task_sampler.kind in ("none", "uniform"):
        logging.info(
            "Task feedback mode for sampler kind=%s is %s (process_count=%d).",
            config.task_sampler.kind,
            task_feedback_mode,
            jax.process_count(),
        )
    task_feedback_is_global_per_step = task_feedback_mode == "global_per_step"
    task_feedback_is_global_per_log = task_feedback_mode == "global_per_log_interval"
    pending_task_indices_local: list[np.ndarray] = []
    pending_losses_local: list[np.ndarray] = []

    step_times = []
    data_times = []
    for step in pbar:
        if isinstance(batch, tuple) and len(batch) == 3:
            observation, actions, task_indices = batch
            batch_for_step = (observation, actions)
        else:
            batch_for_step = batch
            task_indices = None

        step_start = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info, per_example_loss = ptrain_step(train_rng, train_state, batch_for_step)
        if step % config.log_interval == 0:
            jax.block_until_ready(info["loss"])
        step_times.append(time.perf_counter() - step_start)

        if task_indices is not None:
            task_indices_local = np.asarray(task_indices, dtype=np.int64).reshape(-1)
            per_example_loss_local = _to_process_local_numpy(per_example_loss).reshape(-1)
            _assert_task_feedback_compatible(task_indices_local, per_example_loss_local, rank=global_task_rank)

            task_count_sum = int(task_indices_local.shape[0])
            last_task_distribution = task_log_utils.format_task_distribution(task_indices_local)
            if task_feedback_is_global_per_step:
                task_indices_host, per_example_loss_host = _sync_task_feedback(task_indices_local, per_example_loss_local)
                _assert_task_feedback_compatible(task_indices_host, per_example_loss_host, rank=global_task_rank)
                if _task_sampler_supports_update(data_loader):
                    data_loader.update_task_sampler(
                        task_indices_host,
                        per_example_loss_host,
                    )
                _accumulate_task_loss_stats(task_indices_host, per_example_loss_host, task_loss_sums, task_loss_counts)
                task_count_sum = int(task_indices_host.shape[0])
                last_task_distribution = task_log_utils.format_task_distribution(task_indices_host)
            elif task_feedback_is_global_per_log:
                pending_task_indices_local.append(task_indices_local)
                pending_losses_local.append(per_example_loss_local)
            else:
                if _task_sampler_supports_update(data_loader):
                    data_loader.update_task_sampler(
                        task_indices_local,
                        per_example_loss_local,
                    )
                    if (
                        adaptive_feedback_mode == "local_periodic"
                        and ((step + 1) % adaptive_sync_interval == 0 or step == config.num_train_steps - 1)
                    ):
                        _sync_adaptive_task_sampler_state(data_loader)
                _accumulate_task_loss_stats(task_indices_local, per_example_loss_local, task_loss_sums, task_loss_counts)
        else:
            task_count_sum = None

        if dist_ctx.is_primary:
            infos.append(info)
        if step % config.log_interval == 0:
            if task_feedback_is_global_per_log:
                task_indices_local_window = _concat_feedback_chunks(pending_task_indices_local, dtype=np.int64)
                losses_local_window = _concat_feedback_chunks(pending_losses_local, dtype=np.float64)
                task_indices_host_window, losses_host_window = _sync_task_feedback(
                    task_indices_local_window,
                    losses_local_window,
                )
                _assert_task_feedback_compatible(task_indices_host_window, losses_host_window, rank=global_task_rank)
                if task_indices_host_window.size > 0:
                    _accumulate_task_loss_stats(
                        task_indices_host_window,
                        losses_host_window,
                        task_loss_sums,
                        task_loss_counts,
                    )
                    last_task_distribution = task_log_utils.format_task_distribution(task_indices_host_window)
                pending_task_indices_local.clear()
                pending_losses_local.clear()
            if dist_ctx.is_primary:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                reduced_info["step_time_s"] = float(np.mean(step_times))
                reduced_info["data_time_s"] = float(np.mean(data_times)) if data_times else 0.0
                if task_count_sum is not None:
                    if task_feedback_is_global_per_step and task_count_sum != config.batch_size:
                        raise RuntimeError(
                            f"task_count_sum mismatch: expected {config.batch_size}, got {task_count_sum}."
                        )
                    reduced_info["task_count_sum"] = float(task_count_sum)

                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                if last_task_distribution is not None:
                    info_str = f"{info_str}, {last_task_distribution}"
                if task_loss_counts:
                    task_avgs = [
                        (task_id, task_loss_sums[task_id] / count)
                        for task_id, count in task_loss_counts.items()
                        if count > 0
                    ]
                    task_avgs.sort(key=lambda x: x[1], reverse=True)
                    top_parts = [f"{tid}:{loss:.4f}" for tid, loss in task_avgs[:3]]
                    if top_parts:
                        info_str = f"{info_str}, top_task_loss={{ {', '.join(top_parts)} }}"
                pbar.write(f"Step {step}: {info_str}")
                log_payload = dict(reduced_info)
                if task_loss_counts:
                    for task_id, count in task_loss_counts.items():
                        if count > 0:
                            log_payload[f"task_loss/{task_id}"] = task_loss_sums[task_id] / count
                    task_loss_sums.clear()
                    task_loss_counts.clear()
                wandb.log(log_payload, step=step)
                infos = []
            step_times = []
            data_times = []

        if peval_step is not None and val_data_loader is not None and step % config.eval_interval == 0:
            val_metrics = []
            eval_rng = jax.random.fold_in(train_rng, step)
            for val_batch in val_data_loader:
                if isinstance(val_batch, tuple) and len(val_batch) == 3:
                    val_batch = (val_batch[0], val_batch[1])
                eval_rng, step_rng = jax.random.split(eval_rng)
                with sharding.set_mesh(mesh):
                    out = peval_step(step_rng, train_state, val_batch)
                val_metrics.append(out)
            if val_metrics and dist_ctx.is_primary:
                stacked_vals = common_utils.stack_forest(val_metrics)
                reduced_vals = jax.device_get(jax.tree.map(lambda x: jnp.mean(x, axis=0), stacked_vals))
                action_l1_per_dim = np.asarray(reduced_vals.get("action_l1_per_dim", np.array([]))).reshape(-1)
                labels_for_current_eval = _resolve_action_dim_labels(config, len(action_l1_per_dim))
                log_vals, val_str = _format_eval_log_values(
                    reduced_vals,
                    action_dim_labels=labels_for_current_eval,
                )
                pbar.write(f"Step {step}: {val_str}")
                wandb.log(log_vals, step=step)

        data_start = time.perf_counter()
        batch, data_iter = _next_batch(
            data_iter,
            data_loader=data_loader,
            allow_errors=allow_data_errors,
            max_errors=max_data_errors,
            error_state=error_state,
        )
        data_times.append(time.perf_counter() - data_start)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                experiment_yaml=experiment_yaml,
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    _cfg, _yaml_text = load_train_config_from_cli()
    main(_cfg, _yaml_text)
