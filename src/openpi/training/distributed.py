from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import jax
from jax.experimental import multihost_utils
import numpy as np


logger = logging.getLogger(__name__)


_REQUIRED_DISTRIBUTED_ENV = (
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
)

# Fallback environment variables for MPI runtimes (OpenMPI, PMIx, MVAPICH2, Intel MPI).
# Checked in order when the standard variable is not set.
_ENV_FALLBACKS: dict[str, tuple[str, ...]] = {
    "RANK": ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK"),
    "LOCAL_RANK": ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK"),
    "WORLD_SIZE": ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"),
    "LOCAL_WORLD_SIZE": ("OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS", "MV2_COMM_WORLD_LOCAL_SIZE"),
    "MASTER_ADDR": (),
    "MASTER_PORT": (),
}


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    local_rank: int = 0
    local_world_size: int = 1
    master_addr: str = ""
    master_port: int = 0
    rank: int = 0
    world_size: int = 1
    initialized: bool = False

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _get_env(name: str) -> str | None:
    """Get env var with MPI runtime fallbacks."""
    value = os.environ.get(name)
    if value is not None:
        return value
    for fallback in _ENV_FALLBACKS.get(name, ()):
        value = os.environ.get(fallback)
        if value is not None:
            logger.info("Using %s=%s (from %s)", name, value, fallback)
            return value
    return None


def _parse_env() -> DistributedContext | None:
    values: dict[str, str] = {}
    for name in _REQUIRED_DISTRIBUTED_ENV:
        value = _get_env(name)
        if value is None:
            return None
        values[name] = value

    return DistributedContext(
        local_rank=int(values["LOCAL_RANK"]),
        local_world_size=int(values["LOCAL_WORLD_SIZE"]),
        master_addr=values["MASTER_ADDR"],
        master_port=int(values["MASTER_PORT"]),
        rank=int(values["RANK"]),
        world_size=int(values["WORLD_SIZE"]),
        initialized=False,
    )


def _resolve_local_device_ids(
    visible_devices_csv: str | None,
    local_rank: int,
    local_world_size: int,
) -> list[int] | None:
    """Resolve ``local_device_ids`` from ``CUDA_VISIBLE_DEVICES``.

    Handles both integer ordinal and UUID-form device identifiers.
    When multiple local ranks share a node, isolates one device per rank
    by rewriting ``CUDA_VISIBLE_DEVICES`` in the current process.
    """
    if not visible_devices_csv:
        return None

    raw = [d.strip() for d in visible_devices_csv.split(",") if d.strip()]
    if not raw:
        return None

    try:
        int_ids = [int(d) for d in raw]
    except ValueError:
        int_ids = None

    if int_ids is not None:
        logger.info("Using GPUs %s ... ", int_ids)
    else:
        logger.info("CUDA_VISIBLE_DEVICES uses non-integer identifiers: %s", raw)

    if local_world_size > 1:
        # If one ID is already visible, trust external rank-to-GPU assignment.
        if len(raw) == 1:
            return [0]
        # Otherwise isolate one device per local rank and use logical GPU 0.
        if local_rank >= len(raw):
            raise ValueError(
                f"LOCAL_RANK exceeds CUDA_VISIBLE_DEVICES entries: "
                f"local_rank={local_rank}, devices={raw}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = raw[local_rank]
        return [0]

    # Single process per node.
    return int_ids if int_ids is not None else list(range(len(raw)))


_PROXY_ENV_NAMES = (
    "http_proxy", "https_proxy", "no_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
)


def initialize_jax_distributed_from_env() -> DistributedContext:
    """Initialize JAX distributed runtime from standard DDP-like env vars.

    If the required environment variables are not present, returns a single-process
    context without initializing distributed runtime.
    """
    env_ctx = _parse_env()
    if env_ctx is None:
        logger.info("Distributed environment variables not found. Running in single-process mode.")
        return DistributedContext()

    logger.info("Training on %d task(s) ... ", env_ctx.world_size)
    if env_ctx.world_size <= 1:
        return env_ctx

    if jax.distributed.is_initialized():
        return dataclasses.replace(env_ctx, initialized=True)

    # Temporarily remove proxy env vars so gRPC coordinator connections are not proxied.
    proxy_env_values = {name: os.environ.pop(name) for name in _PROXY_ENV_NAMES if name in os.environ}

    local_device_ids = _resolve_local_device_ids(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        env_ctx.local_rank,
        env_ctx.local_world_size,
    )

    init_kwargs = dict(
        coordinator_address=f"{env_ctx.master_addr}:{env_ctx.master_port}",
        num_processes=env_ctx.world_size,
        process_id=env_ctx.rank,
        local_device_ids=local_device_ids,
    )
    logger.info("Initializing w/ %s ... ", init_kwargs)
    try:
        jax.distributed.initialize(**init_kwargs)
    finally:
        for name, value in proxy_env_values.items():
            os.environ[name] = value

    return dataclasses.replace(env_ctx, initialized=True)


def allgather_processes_1d(values: np.ndarray) -> np.ndarray:
    """All-gather a 1D array across processes and concatenate results."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"allgather_processes_1d expects a 1D array, got shape {values.shape}.")
    if jax.process_count() <= 1:
        return values
    gathered = multihost_utils.process_allgather(values, tiled=True)
    return np.asarray(gathered).reshape(-1)


def allreduce_mean_processes_1d(values: np.ndarray) -> np.ndarray:
    """All-reduce(mean) a 1D array across processes."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"allreduce_mean_processes_1d expects a 1D array, got shape {values.shape}.")
    if jax.process_count() <= 1:
        return values
    gathered = np.asarray(multihost_utils.process_allgather(values, tiled=False), dtype=np.float64)
    if gathered.ndim != 2:
        raise RuntimeError(f"Expected gathered array with shape (world_size, n), got {gathered.shape}.")
    return gathered.mean(axis=0)


def broadcast_one_to_all(values: Any, *, is_source: bool) -> Any:
    """Broadcast a pytree from the source process to all processes."""
    if jax.process_count() <= 1:
        return values
    return multihost_utils.broadcast_one_to_all(values, is_source=is_source)


def broadcast_episode_split(
    train_episodes: list[int] | None,
    val_episodes: list[int] | None,
    *,
    is_source: bool,
) -> tuple[list[int] | None, list[int] | None]:
    """Broadcast train/val episode splits from rank0 to all processes."""
    if jax.process_count() <= 1:
        return train_episodes, val_episodes

    if is_source:
        train_len = len(train_episodes) if train_episodes is not None else -1
        val_len = len(val_episodes) if val_episodes is not None else -1
        lens = np.asarray([train_len, val_len], dtype=np.int64)
    else:
        lens = np.asarray([0, 0], dtype=np.int64)
    lens = np.asarray(broadcast_one_to_all(lens, is_source=is_source), dtype=np.int64)

    train_len = int(lens[0])
    val_len = int(lens[1])

    if train_len >= 0:
        if is_source:
            train_arr = np.asarray(train_episodes, dtype=np.int64)
        else:
            train_arr = np.empty(train_len, dtype=np.int64)
        train_arr = np.asarray(broadcast_one_to_all(train_arr, is_source=is_source), dtype=np.int64)
        train_episodes = train_arr.tolist()
    else:
        train_episodes = None

    if val_len >= 0:
        if is_source:
            val_arr = np.asarray(val_episodes, dtype=np.int64)
        else:
            val_arr = np.empty(val_len, dtype=np.int64)
        val_arr = np.asarray(broadcast_one_to_all(val_arr, is_source=is_source), dtype=np.int64)
        val_episodes = val_arr.tolist()
    else:
        val_episodes = None

    return train_episodes, val_episodes
