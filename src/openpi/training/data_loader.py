from collections.abc import Iterator, Sequence
import inspect
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.fast_lerobot_dataset import FastLeRobotDataset
from openpi.training.fast_lerobot_metadata import FastLeRobotDatasetMetadata
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.task_sampler as _task_sampler
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        passthrough_keys: Sequence[str] = (),
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._passthrough_keys = tuple(passthrough_keys)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        sample = self._dataset[index]
        transformed = self._transform(sample)
        if self._passthrough_keys and isinstance(sample, dict) and isinstance(transformed, dict):
            for key in self._passthrough_keys:
                if key in sample and key not in transformed:
                    transformed[key] = sample[key]
        return transformed

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _infer_fast_lerobot_include_columns(data_config: _config.DataConfig) -> list[str] | None:
    """Infer a minimal set of parquet columns required by the training pipeline.

    We only infer this when RepackTransform is present; otherwise we leave the loader
    behavior unchanged to avoid accidentally dropping required dataset-specific keys.
    """
    source_keys: set[str] = set()
    for transform in data_config.repack_transforms.inputs:
        if not isinstance(transform, _transforms.RepackTransform):
            continue
        for source_key in jax.tree.leaves(transform.structure):
            if isinstance(source_key, str) and source_key:
                source_keys.add(source_key)

    if not source_keys:
        return None

    include_keys = set(source_keys)
    include_keys.update(str(key) for key in data_config.action_sequence_keys)
    # Required for FastLeRobotDataset item assembly and downstream task split/sampling.
    include_keys.update(("timestamp", "task_index"))
    return sorted(include_keys)


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    episodes: Sequence[int] | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if data_config.fast_lerobot:
        dataset_meta = FastLeRobotDatasetMetadata(repo_id)
    else:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset_cls = lerobot_dataset.LeRobotDataset
    dataset_kwargs = {}
    if data_config.fast_lerobot:
        include_columns = _infer_fast_lerobot_include_columns(data_config)
        if include_columns is not None:
            logging.info(
                "Fast LeRobot parquet projection enabled: %s",
                include_columns,
            )
        dataset_cls = FastLeRobotDataset
        dataset_kwargs = {
            "lerobot_backend": data_config.lerobot_backend,
            "lerobot_checks": data_config.lerobot_checks,
            "include_columns": include_columns,
        }
    dataset = dataset_cls(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        tolerance_s=1e-3, # default is 1e-4
        video_backend=data_config.video_backend or "pyav",
        episodes=episodes,
        **dataset_kwargs,
    )
    if episodes is not None and not data_config.fast_lerobot:
        # LeRobotDataset expects episode_index values to be contiguous when using episodes filtering.
        # We remap the episode_data_index to the original episode indices to avoid out-of-bounds.
        lengths = []
        for ep_idx in episodes:
            episode = dataset.meta.episodes.get(ep_idx)
            if episode is None:
                raise KeyError(f"episode_index {ep_idx} missing from metadata.")
            lengths.append(int(episode["length"]))
        cumulative = np.cumsum(lengths)
        max_ep = max(episodes) if episodes else 0
        from_arr = np.zeros(max_ep + 1, dtype=np.int64)
        to_arr = np.zeros(max_ep + 1, dtype=np.int64)
        start = 0
        for ep_idx, end in zip(episodes, cumulative, strict=True):
            from_arr[ep_idx] = start
            to_arr[ep_idx] = end
            start = end
        dataset.episode_data_index = {
            "from": torch.LongTensor(from_arr),
            "to": torch.LongTensor(to_arr),
        }

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        passthrough_keys=("task_index",),
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    episodes: Sequence[int] | None = None,
    enable_task_sampler: bool = True,
    include_task_index: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        enable_task_sampler: Whether to enable task sampling from config.task_sampler.
        include_task_index: Whether to append task indices to each yielded batch.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")
    task_sampler_config = config.task_sampler if enable_task_sampler else None

    if data_config.rlds_data_dir is not None:
        if task_sampler_config is not None and task_sampler_config.kind != "none":
            raise NotImplementedError("Task sampling is not supported for RLDS datasets.")
        if episodes is not None:
            raise ValueError("Episode filtering is not supported for RLDS data loaders.")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        task_sampler_config=task_sampler_config,
        episodes=episodes,
        include_task_index=include_task_index,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    seed: int = 0,
    framework: str = "jax",
    task_sampler_config: _config.TaskSamplerConfig | None = None,
    episodes: Sequence[int] | None = None,
    include_task_index: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, episodes=episodes)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    task_sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if task_sampler_config is not None and task_sampler_config.kind != "none":
                raise NotImplementedError("Task sampling is not supported with PyTorch DDP.")
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
        if (
            jax.process_count() > 1
            and (task_sampler_config is None or task_sampler_config.kind == "none")
        ):
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=jax.process_count(),
                rank=jax.process_index(),
                shuffle=shuffle,
                drop_last=True,
            )

    if sampler is None and task_sampler_config is not None and task_sampler_config.kind != "none":
        task_sampler = _task_sampler.create_task_sampler(
            task_sampler_config,
            dataset,
            seed=seed,
            num_replicas=jax.process_count() if framework != "pytorch" else 1,
            rank=jax.process_index() if framework != "pytorch" else 0,
            global_batch_size=batch_size,
            local_batch_size=local_batch_size,
        )
        sampler = task_sampler

    logging.info(f"local_batch_size: {local_batch_size}")
    loader_kwargs = {
        "local_batch_size": local_batch_size,
        "sharding": None if framework == "pytorch" else sharding,
        "shuffle": (sampler is None and shuffle),  # Don't shuffle if using sampler
        "sampler": sampler,
        "num_batches": num_batches,
        "num_workers": num_workers,
        "seed": seed,
        "framework": framework,
    }
    supported_loader_args = inspect.signature(TorchDataLoader.__init__).parameters
    if "pin_memory" in supported_loader_args:
        loader_kwargs["pin_memory"] = pin_memory
    elif pin_memory:
        logging.warning("TorchDataLoader does not support pin_memory; ignoring the configured value.")
    if "prefetch_factor" in supported_loader_args:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    elif prefetch_factor != 2:
        logging.warning("TorchDataLoader does not support prefetch_factor; ignoring the configured value.")

    data_loader = TorchDataLoader(dataset, **loader_kwargs)

    return DataLoaderImpl(data_config, data_loader, task_sampler=task_sampler, include_task_index=include_task_index)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._sampler = sampler
        self._epoch = 0

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed + jax.process_index())
        loader_kwargs = {
            "batch_size": local_batch_size,
            "shuffle": (sampler is None and shuffle),
            "sampler": sampler,
            "num_workers": num_workers,
            "multiprocessing_context": mp_context,
            "persistent_workers": num_workers > 0,
            "collate_fn": _collate_fn,
            "worker_init_fn": _worker_init_fn,
            "drop_last": True,
            "generator": generator,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            **loader_kwargs,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            sampler = getattr(self, "_sampler", None)
            epoch = getattr(self, "_epoch", 0)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
                self._epoch = epoch + 1
                if hasattr(sampler, "reset_queue"):
                    sampler.reset_queue()
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _infer_process_local_batch_size(actions) -> int:
    """Infer process-local batch size from possibly global-sharded arrays."""
    if hasattr(actions, "addressable_shards") and not getattr(actions, "is_fully_addressable", True):
        local_batch = 0
        for shard in actions.addressable_shards:
            shard_data = np.asarray(shard.data)
            if shard_data.ndim == 0:
                continue
            local_batch += int(shard_data.shape[0])
        if local_batch > 0:
            return local_batch

    return int(actions.shape[0])


def _to_process_local_1d(values) -> np.ndarray:
    """Convert possibly global sharded values into process-local 1D ndarray."""
    if hasattr(values, "addressable_shards") and not getattr(values, "is_fully_addressable", True):
        shards = sorted(
            values.addressable_shards,
            key=lambda shard: tuple(
                idx if isinstance(idx, int) else (0 if idx.start is None else idx.start)
                for idx in shard.index
            ),
        )
        if not shards:
            return np.asarray([], dtype=np.int64)
        return np.concatenate([np.asarray(shard.data).reshape(-1) for shard in shards], axis=0)
    return np.asarray(values).reshape(-1)


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
        *,
        task_sampler: _task_sampler.TaskSamplerBase | None = None,
        include_task_index: bool = False,
    ):
        self._data_config = data_config
        self._data_loader = data_loader
        self._task_sampler = task_sampler
        self._include_task_index = include_task_index
        self._warned_missing_task_index = False

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def update_task_sampler(self, task_indices: np.ndarray, losses: np.ndarray) -> None:
        if self._task_sampler is None:
            return
        if hasattr(self._task_sampler, "update"):
            self._task_sampler.update(task_indices, losses)

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            actions = batch["actions"]
            if self._include_task_index:
                if self._task_sampler is not None:
                    batch_size = _infer_process_local_batch_size(actions)
                    if batch_size <= 0:
                        raise _task_sampler.TaskSamplerInvariantError(
                            "Process-local batch size must be positive for task sampling, "
                            f"got {batch_size}."
                        )
                    task_indices = self._task_sampler.pop_task_indices(batch_size)
                    yield observation, actions, task_indices
                    continue

                if "task_index" in batch:
                    task_indices = _to_process_local_1d(batch["task_index"]).astype(np.int64)
                    yield observation, actions, task_indices
                    continue

                if not self._warned_missing_task_index:
                    logging.warning(
                        "Task-index logging requested, but `task_index` is missing from batches. "
                        "Per-task loss logging will be skipped for this loader."
                    )
                    self._warned_missing_task_index = True

            yield observation, actions
