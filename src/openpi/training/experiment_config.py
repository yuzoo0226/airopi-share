"""
YAML設定から動的にTrainConfigを生成するモジュール

使用方法:
    from openpi.training.experiment_config import ExperimentConfig
    
    config = ExperimentConfig.from_yaml("configs/experiments/my_exp.yaml")
    train_config = config.to_train_config()
"""

import dataclasses
import logging
import math
import os
import pathlib
import re
from typing import Any, Literal, Optional

import yaml

import openpi.models.finetune as _finetune
import openpi.models.pi0_config as pi0_config
import openpi.policies.hsr_policy as hsr_policy
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


logger = logging.getLogger(__name__)


def _read_env_int(name: str) -> int | None:
    """Read first integer token from env var."""
    value = os.environ.get(name)
    if not value:
        return None
    match = re.search(r"\d+", value)
    if match is None:
        return None
    return int(match.group())


def _count_visible_cuda_devices() -> int | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        return None
    devices = [token.strip() for token in value.split(",") if token.strip()]
    if not devices:
        return None
    return len(devices)


def deep_merge(base: dict, override: dict) -> dict:
    """ネストされた辞書を再帰的にマージする"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_with_inheritance(yaml_path: pathlib.Path) -> dict:
    """_base_フィールドを解決しながらYAMLを読み込む"""
    with open(yaml_path) as f:
        config = yaml.safe_load(f) or {}
    
    # ベース設定の読み込み
    if "_base_" in config:
        base_path = yaml_path.parent / config.pop("_base_")
        base_config = load_yaml_with_inheritance(base_path.resolve())
        config = deep_merge(base_config, config)
    
    return config


@dataclasses.dataclass
class GPUConfig:
    """GPU設定とプロセスローカル情報の計算"""
    num_gpus: int = 1
    base_gpus: int = 1
    fsdp_devices: int = 1
    
    @classmethod
    def from_dict(cls, d: dict) -> "GPUConfig":
        return cls(
            num_gpus=d.get("num_gpus", 1),
            base_gpus=d.get("base_gpus", 1),
            fsdp_devices=d.get("fsdp_devices", 1),
        )
    
    @classmethod
    def auto_detect(cls) -> "GPUConfig":
        """環境から自動検出"""
        # JAX/CUDAからGPU数を取得
        try:
            import jax
            num_gpus = jax.device_count()
        except:
            num_gpus = int(os.environ.get("NUM_GPUS", "1"))
        
        return cls(num_gpus=num_gpus)
    
    @property
    def scale_factor(self) -> float:
        """batch_sizeのスケーリング係数"""
        return self.num_gpus / self.base_gpus
    
    @property
    def lr_scale_factor(self) -> float:
        """learning rateのスケーリング係数 (sqrt rule)"""
        return math.sqrt(self.scale_factor)

    def local_gpus_per_process(self) -> int:
        """Estimate GPUs used by each training process."""
        visible_cuda_devices = _count_visible_cuda_devices()
        local_world_size = _read_env_int("LOCAL_WORLD_SIZE")
        if visible_cuda_devices is not None:
            if local_world_size and local_world_size > 0:
                return max(1, visible_cuda_devices // local_world_size)
            return max(1, visible_cuda_devices)

        slurm_gpus_per_task = _read_env_int("SLURM_GPUS_PER_TASK")
        if slurm_gpus_per_task and slurm_gpus_per_task > 0:
            return slurm_gpus_per_task

        slurm_gpus_on_node = _read_env_int("SLURM_GPUS_ON_NODE")
        slurm_tasks_per_node = _read_env_int("SLURM_NTASKS_PER_NODE")
        if (
            slurm_gpus_on_node is not None
            and slurm_gpus_on_node > 0
            and slurm_tasks_per_node is not None
            and slurm_tasks_per_node > 0
        ):
            return max(1, slurm_gpus_on_node // slurm_tasks_per_node)

        world_size = _read_env_int("WORLD_SIZE")
        if world_size is not None and world_size > 0:
            return max(1, self.num_gpus // world_size)

        openpi_num_nodes = _read_env_int("OPENPI_NUM_NODES")
        if openpi_num_nodes is not None and openpi_num_nodes > 0:
            return max(1, self.num_gpus // openpi_num_nodes)

        return max(1, self.num_gpus)


@dataclasses.dataclass
class LRScheduleConfig:
    """学習率スケジュール設定"""
    type: str = "cosine_decay"
    warmup_steps: int = 1000
    peak_lr: float = 5.0e-5
    decay_lr: float = 5.0e-6
    decay_steps: Optional[int] = None  # Noneの場合num_train_stepsを使用
    
    @classmethod
    def from_dict(cls, d: dict) -> "LRScheduleConfig":
        return cls(
            type=d.get("type", "cosine_decay"),
            warmup_steps=d.get("warmup_steps", 1000),
            peak_lr=d.get("peak_lr", 5.0e-5),
            decay_lr=d.get("decay_lr", 5.0e-6),
            decay_steps=d.get("decay_steps"),
        )
    
    def scale(self, batch_scale_factor: float | GPUConfig) -> "LRScheduleConfig":
        """有効batch size比に応じてスケーリング (sqrt rule)。"""
        # Backward compatibility: accept GPUConfig and convert to ratio.
        if isinstance(batch_scale_factor, GPUConfig):
            batch_scale_factor = batch_scale_factor.scale_factor
        else:
            batch_scale_factor = float(batch_scale_factor)
        if batch_scale_factor <= 0:
            raise ValueError("batch_scale_factor must be > 0.")
        lr_scale_factor = math.sqrt(batch_scale_factor)
        return dataclasses.replace(
            self,
            peak_lr=self.peak_lr * lr_scale_factor,
            decay_lr=self.decay_lr * lr_scale_factor,
        )
    
    def to_optimizer_config(self, num_train_steps: int) -> _optimizer.LRScheduleConfig:
        """openpiのLRScheduleConfigに変換"""
        decay_steps = self.decay_steps or num_train_steps
        return _optimizer.CosineDecaySchedule(
            warmup_steps=self.warmup_steps,
            peak_lr=self.peak_lr,
            decay_steps=decay_steps,
            decay_lr=self.decay_lr,
        )


@dataclasses.dataclass
class TrainingConfig:
    """学習設定"""
    batch_size: int = 128
    num_workers: Optional[int] = None
    pin_memory: bool = False
    prefetch_factor: int = 2
    num_train_steps: int = 10000
    val_split_fraction: Optional[float] = 0.1
    val_split_seed: int = 42
    eval_interval: Optional[int] = 1000
    eval_num_batches: int = 20
    eval_num_sample_steps: int = 10
    lr_schedule: LRScheduleConfig = dataclasses.field(default_factory=LRScheduleConfig)
    save_interval: int = 1000
    keep_period: int = 5000
    log_interval: int = 100
    
    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        lr_dict = d.get("lr_schedule", {})
        return cls(
            batch_size=d.get("batch_size", 128),
            num_workers=d.get("num_workers"),
            pin_memory=d.get("pin_memory", False),
            prefetch_factor=d.get("prefetch_factor", 2),
            num_train_steps=d.get("num_train_steps", 10000),
            val_split_fraction=d.get("val_split_fraction", 0.1),
            val_split_seed=d.get("val_split_seed", 42),
            eval_interval=d.get("eval_interval", 1000),
            eval_num_batches=d.get("eval_num_batches", 20),
            eval_num_sample_steps=d.get("eval_num_sample_steps", 10),
            lr_schedule=LRScheduleConfig.from_dict(lr_dict),
            save_interval=d.get("save_interval", 1000),
            keep_period=d.get("keep_period", 5000),
            log_interval=d.get("log_interval", 100),
        )
    
    def scale(self, gpu_config: GPUConfig, scale_config: dict) -> "TrainingConfig":
        """スケーリング設定を適用する。

        scale_learning_rate / scale_train_steps は GPU 数ではなく
        `effective_batch_size / base_batch_size` の比率で計算する。
        """
        result = dataclasses.replace(self)
        
        if scale_config.get("scale_batch_size", True):
            result = dataclasses.replace(
                result,
                batch_size=int(self.batch_size * gpu_config.scale_factor),
            )

        base_batch_size = scale_config.get("base_batch_size", self.batch_size)
        try:
            base_batch_size = float(base_batch_size)
        except (TypeError, ValueError) as e:
            raise ValueError("scaling.base_batch_size must be numeric.") from e
        if base_batch_size <= 0:
            raise ValueError("scaling.base_batch_size must be > 0.")
        batch_scale_factor = result.batch_size / base_batch_size
        
        if scale_config.get("scale_learning_rate", True):
            result = dataclasses.replace(
                result,
                lr_schedule=self.lr_schedule.scale(batch_scale_factor),
            )
        
        if scale_config.get("scale_train_steps", True):
            result = dataclasses.replace(
                result,
                num_train_steps=max(1, int(self.num_train_steps / batch_scale_factor)),
            )
        return result

    def resolve_num_workers(self, gpu_config: GPUConfig) -> "TrainingConfig":
        if self.num_workers is not None:
            return self
        return dataclasses.replace(self, num_workers=gpu_config.local_gpus_per_process())


@dataclasses.dataclass
class DatasetConfig:
    """データセット設定"""
    name: str = ""
    repo_id: str = ""
    data_dir: str = ""
    asset_id: Optional[str] = None
    assets_dir: Optional[str] = None
    prompt_from_task: bool = True
    action_mode: str = "relative"
    adapt_to_pi: bool = True
    convert_gripper: bool = False
    base_action_dim: int = 3
    fast_lerobot: bool = False
    lerobot_backend: Literal["parquet", "hf"] = "parquet"
    lerobot_checks: Literal["all", "init_only", "none"] = "none"
    video_backend: Optional[str] = None
    action_sequence_keys: tuple = ("action.relative",)
    
    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        action_keys = d.get("action_sequence_keys", ["action.relative"])
        if isinstance(action_keys, list):
            action_keys = tuple(action_keys)
        return cls(
            name=d.get("name", ""),
            repo_id=d.get("repo_id", ""),
            data_dir=d.get("data_dir", os.environ.get("DATA_DIR", "")),
            asset_id=d.get("asset_id"),
            assets_dir=d.get("assets_dir"),
            prompt_from_task=d.get("prompt_from_task", True),
            action_mode=d.get("action_mode", "relative"),
            adapt_to_pi=d.get("adapt_to_pi", True),
            convert_gripper=d.get("convert_gripper", False),
            base_action_dim=d.get("base_action_dim", 3),
            fast_lerobot=d.get("fast_lerobot", False),
            lerobot_backend=d.get("lerobot_backend", "parquet"),
            lerobot_checks=d.get("lerobot_checks", "none"),
            video_backend=d.get("video_backend"),
            action_sequence_keys=action_keys,
        )


@dataclasses.dataclass
class ModelConfig:
    """モデル設定"""
    type: str = "pi0"  # pi0 or pi05
    variant: str = "lora"  # lora or full
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    finetune_recipe: _finetune.FineTuneRecipeConfig | None = None
    image_encoder_mode: pi0_config.ImageEncoderMode = "shared"
    action_dim: int = 32
    action_horizon: int = 16
    # Kept for backward compatibility. LoRA freezing is inferred from model variants.
    freeze_action_expert: bool = True
    ema_decay: Optional[float] = None
    
    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        model_type = d.get("type", "pi0")
        explicit_variant = d.get("variant")
        explicit_paligemma_variant = d.get("paligemma_variant")
        explicit_action_expert_variant = d.get("action_expert_variant")
        finetune_recipe = _finetune.FineTuneRecipeConfig.from_dict(d.get("finetune_recipe"))

        # Keep pi0 defaulting to LoRA, but preserve historical pi05 default behavior
        # (full fine-tuning) unless YAML explicitly sets LoRA variants.
        default_variant = "full" if model_type == "pi05" or finetune_recipe is not None else "lora"
        inferred_variant = default_variant
        if finetune_recipe is None and (explicit_paligemma_variant is not None or explicit_action_expert_variant is not None):
            uses_lora_in_variants = ("lora" in (explicit_paligemma_variant or "")) or (
                "lora" in (explicit_action_expert_variant or "")
            )
            inferred_variant = "lora" if uses_lora_in_variants else "full"
        variant = explicit_variant if explicit_variant is not None else inferred_variant

        if finetune_recipe is not None:
            if variant == "lora":
                raise ValueError("model.variant='lora' cannot be combined with model.finetune_recipe.")
            if "lora" in (explicit_paligemma_variant or "") or "lora" in (explicit_action_expert_variant or ""):
                raise ValueError(
                    "LoRA model variants cannot be combined with model.finetune_recipe. "
                    "Use full Gemma variants with finetune_recipe.vision_lora instead."
                )

        if variant == "lora":
            default_paligemma_variant = "gemma_2b_lora"
            default_action_expert_variant = "gemma_300m_lora"
        else:
            default_paligemma_variant = "gemma_2b"
            default_action_expert_variant = "gemma_300m"

        return cls(
            type=model_type,
            variant=variant,
            paligemma_variant=d.get("paligemma_variant", default_paligemma_variant),
            action_expert_variant=d.get("action_expert_variant", default_action_expert_variant),
            finetune_recipe=finetune_recipe,
            image_encoder_mode=d.get("image_encoder_mode", "shared"),
            action_dim=d.get("action_dim", 32),
            action_horizon=d.get("action_horizon", 16),
            freeze_action_expert=d.get("freeze_action_expert", True),
            ema_decay=d.get("ema_decay"),
        )
    
    @property
    def is_pi05(self) -> bool:
        return self.type == "pi05"

    @property
    def uses_lora(self) -> bool:
        return (
            (self.finetune_recipe is not None and self.finetune_recipe.vision_lora.enabled)
            or ("lora" in self.paligemma_variant)
            or ("lora" in self.action_expert_variant)
        )
    
    def to_pi0_config(self) -> pi0_config.Pi0Config:
        """openpiのPi0Configに変換"""
        if self.is_pi05:
            return pi0_config.Pi0Config(
                pi05=True,
                action_dim=self.action_dim,
                action_horizon=self.action_horizon,
                paligemma_variant=self.paligemma_variant,
                action_expert_variant=self.action_expert_variant,
                finetune_recipe=self.finetune_recipe,
                image_encoder_mode=self.image_encoder_mode,
            )
        else:
            return pi0_config.Pi0Config(
                paligemma_variant=self.paligemma_variant,
                action_expert_variant=self.action_expert_variant,
                finetune_recipe=self.finetune_recipe,
                image_encoder_mode=self.image_encoder_mode,
            )


@dataclasses.dataclass
class WeightLoaderConfig:
    """重みローダー設定."""

    type: str = "checkpoint"
    params_path: Optional[str] = None

    @staticmethod
    def normalize_type(value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "checkpoint": "checkpoint",
            "checkpointweightloader": "checkpoint",
            "checkpoint_weight_loader": "checkpoint",
            "paligemma": "paligemma",
            "paligemmaweightloader": "paligemma",
            "paligemma_weight_loader": "paligemma",
            "noop": "noop",
            "none": "noop",
            "no_op": "noop",
            "noopweightloader": "noop",
            "noop_weight_loader": "noop",
        }
        if normalized not in aliases:
            raise ValueError(
                "Unsupported weight_loader.type: "
                f"{value!r}. Supported: checkpoint, paligemma, noop."
            )
        return aliases[normalized]

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, *, default_checkpoint_path: Optional[str]) -> "WeightLoaderConfig":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise ValueError("weight_loader must be a mapping.")

        loader_type = cls.normalize_type(str(d.get("type", "checkpoint")))
        params_path = d.get("params_path")
        if loader_type == "checkpoint" and params_path is None:
            params_path = default_checkpoint_path
        return cls(type=loader_type, params_path=params_path)

    def build(self) -> weight_loaders.WeightLoader:
        if self.type == "paligemma":
            return weight_loaders.PaliGemmaWeightLoader()
        if self.type == "noop":
            return weight_loaders.NoOpWeightLoader()
        params_path = self.params_path or "gs://openpi-assets/checkpoints/pi0_base/params"
        return weight_loaders.CheckpointWeightLoader(params_path)


@dataclasses.dataclass
class ExperimentConfig:
    """実験設定の統合クラス"""
    name: str
    project_name: str = "openpi"
    seed: int = 42
    wandb_enabled: bool = True
    
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    gpu: GPUConfig = dataclasses.field(default_factory=GPUConfig)
    scaling: dict = dataclasses.field(default_factory=dict)
    task_sampler: dict = dataclasses.field(default_factory=dict)
    weight_loader: WeightLoaderConfig = dataclasses.field(default_factory=WeightLoaderConfig)
    
    # チェックポイント設定
    checkpoint_base_dir: str = ""
    assets_base_dir: str = "./assets"
    base_checkpoint_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str | pathlib.Path, auto_detect_gpu: bool = True) -> "ExperimentConfig":
        """YAMLファイルから設定を読み込む"""
        yaml_path = pathlib.Path(yaml_path)
        config_dict = load_yaml_with_inheritance(yaml_path)
        
        # 環境変数からのオーバーライド
        env_overrides = {
            "checkpoint_base_dir": os.environ.get("CHECKPOINT_DIR", "./checkpoints"),
            "data_dir": os.environ.get("DATA_DIR", ""),
        }
        
        experiment = config_dict.get("experiment", {})
        dataset_dict = config_dict.get("dataset", {})
        if not dataset_dict.get("data_dir"):
            dataset_dict["data_dir"] = env_overrides["data_dir"]
        
        # GPU設定
        gpu_dict = config_dict.get("gpu", {})
        if auto_detect_gpu:
            detected_gpu = GPUConfig.auto_detect()
            # YAMLのnum_gpusが明示的に設定されていなければ自動検出を使用
            if "num_gpus" not in gpu_dict or gpu_dict.get("auto_detect", True):
                gpu_dict["num_gpus"] = detected_gpu.num_gpus
        
        gpu_config = GPUConfig.from_dict(gpu_dict)
        
        # プリセットの適用
        presets = config_dict.get("presets", {})
        preset_key = f"multi_gpu_{gpu_config.num_gpus}" if gpu_config.num_gpus > 1 else "single_gpu"
        if preset_key in presets:
            preset = presets[preset_key]
            # プリセットをマージ
            training_dict = deep_merge(
                config_dict.get("training", {}),
                preset.get("training", {})
            )
        else:
            training_dict = config_dict.get("training", {})
        
        # 各コンポーネントの作成
        dataset_config = DatasetConfig.from_dict(dataset_dict)
        model_config = ModelConfig.from_dict(config_dict.get("model", {}))
        training_config = TrainingConfig.from_dict(training_dict)
        scaling_config = config_dict.get("scaling", {})
        
        # スケーリング適用（プリセットが使われていない場合）
        if preset_key not in presets and gpu_config.num_gpus > 1:
            training_config = training_config.scale(gpu_config, scaling_config)
        training_config = training_config.resolve_num_workers(gpu_config)
        
        # ベースチェックポイントのパス
        checkpoints_cfg = config_dict.get("checkpoints", {})
        checkpoints = checkpoints_cfg.get("base_model", {})
        base_checkpoint = checkpoints.get(model_config.type)
        weight_loader_dict = config_dict.get("weight_loader")
        if weight_loader_dict is None:
            weight_loader_dict = checkpoints_cfg.get("weight_loader")
        weight_loader_config = WeightLoaderConfig.from_dict(
            weight_loader_dict, default_checkpoint_path=base_checkpoint
        )
        
        return cls(
            name=experiment.get("name", yaml_path.stem),
            project_name=experiment.get("project_name", "openpi"),
            seed=experiment.get("seed", 42),
            wandb_enabled=experiment.get("wandb_enabled", True),
            dataset=dataset_config,
            model=model_config,
            training=training_config,
            gpu=gpu_config,
            scaling=scaling_config,
            task_sampler=config_dict.get("task_sampler", {}),
            weight_loader=weight_loader_config,
            checkpoint_base_dir=env_overrides["checkpoint_base_dir"],
            assets_base_dir=config_dict.get("assets_base_dir", "./assets"),
            base_checkpoint_path=base_checkpoint,
        )
    
    def to_train_config(self) -> _config.TrainConfig:
        """openpiのTrainConfigに変換"""
        # モデル設定
        pi0_config_obj = self.model.to_pi0_config()
        
        # データ設定
        base_data_config = _config.DataConfig(
            prompt_from_task=self.dataset.prompt_from_task,
            fast_lerobot=self.dataset.fast_lerobot,
            lerobot_backend=self.dataset.lerobot_backend,
            lerobot_checks=self.dataset.lerobot_checks,
            video_backend=self.dataset.video_backend,
        )
        if self.model.is_pi05:
            assets_config = _config.AssetsConfig(
                assets_dir=self.dataset.assets_dir or f"./assets/{self.name}",
                asset_id=self.dataset.asset_id or self.dataset.repo_id,
            )
            data_config = _config.LeRobotHSRDataConfig(
                repo_id=self.dataset.repo_id,
                assets=assets_config,
                base_config=base_data_config,
                action_mode=self.dataset.action_mode,
                adapt_to_pi=self.dataset.adapt_to_pi,
                convert_gripper=self.dataset.convert_gripper,
                base_action_dim=self.dataset.base_action_dim,
            )
        else:
            data_config = _config.LeRobotHSRDataConfig(
                repo_id=self.dataset.repo_id,
                base_config=base_data_config,
                action_mode=self.dataset.action_mode,
                adapt_to_pi=self.dataset.adapt_to_pi,
                convert_gripper=self.dataset.convert_gripper,
                base_action_dim=self.dataset.base_action_dim,
            )
        
        # Weight loader
        weight_loader = self.weight_loader.build()
        
        # Freeze filter
        freeze_filter = pi0_config_obj.get_freeze_filter()
        
        # LR Schedule
        lr_schedule = self.training.lr_schedule.to_optimizer_config(self.training.num_train_steps)
        task_sampler_cfg = _config.TaskSamplerConfig(**self.task_sampler)
        
        train_config = _config.TrainConfig(
            name=self.name,
            project_name=self.project_name,
            exp_name=self.name,
            model=pi0_config_obj,
            data=data_config,
            weight_loader=weight_loader,
            lr_schedule=lr_schedule,
            batch_size=self.training.batch_size,
            num_workers=(
                self.training.num_workers
                if self.training.num_workers is not None
                else self.gpu.local_gpus_per_process()
            ),
            pin_memory=self.training.pin_memory,
            prefetch_factor=self.training.prefetch_factor,
            num_train_steps=self.training.num_train_steps,
            save_interval=self.training.save_interval,
            keep_period=self.training.keep_period,
            log_interval=self.training.log_interval,
            seed=self.seed,
            wandb_enabled=self.wandb_enabled,
            checkpoint_base_dir=self.checkpoint_base_dir,
            assets_base_dir=self.assets_base_dir,
            ema_decay=self.model.ema_decay,
            fsdp_devices=self.gpu.fsdp_devices,
            task_sampler=task_sampler_cfg,
            val_split_fraction=self.training.val_split_fraction,
            val_split_seed=self.training.val_split_seed,
            eval_interval=self.training.eval_interval,
            eval_num_batches=self.training.eval_num_batches,
            eval_num_sample_steps=self.training.eval_num_sample_steps,
        )
        
        return dataclasses.replace(train_config, freeze_filter=freeze_filter)
    
    def summary(self) -> str:
        """設定の要約を出力"""
        lines = [
            f"=== Experiment Configuration ===",
            f"Name: {self.name}",
            f"Model: {self.model.type} ({self.model.variant})",
            f"  finetune_recipe: {'enabled' if self.model.finetune_recipe is not None else 'disabled'}",
            f"  image_encoder_mode: {self.model.image_encoder_mode}",
            (
                f"  vision_train_mode: {self.model.finetune_recipe.vision_train_mode}"
                if self.model.finetune_recipe is not None
                else ""
            ),
            f"",
            f"Dataset:",
            f"  repo_id: {self.dataset.repo_id}",
            f"  data_dir: {self.dataset.data_dir}",
            f"",
            f"Training:",
            f"  batch_size: {self.training.batch_size}",
            f"  num_train_steps: {self.training.num_train_steps}",
            f"  peak_lr: {self.training.lr_schedule.peak_lr:.2e}",
            f"  decay_lr: {self.training.lr_schedule.decay_lr:.2e}",
            f"  num_workers: {self.training.num_workers}",
            f"",
            f"GPU:",
            f"  num_gpus: {self.gpu.num_gpus}",
            f"  fsdp_devices: {self.gpu.fsdp_devices}",
            f"================================",
        ]
        return "\n".join(lines)


def load_experiment_config(yaml_path: str | pathlib.Path) -> _config.TrainConfig:
    """YAMLファイルからTrainConfigを直接生成するヘルパー関数"""
    exp_config = ExperimentConfig.from_yaml(yaml_path)
    logger.info(exp_config.summary())
    return exp_config.to_train_config()


def dump_merged_yaml(yaml_path: str | pathlib.Path) -> str:
    """`_base_` 継承を解決した1枚の YAML 文字列を返す。

    checkpoint に同梱するための「自己完結 YAML」を生成する用途。
    元ファイルのパスをヘッダコメントとして残す。
    """
    path = pathlib.Path(yaml_path)
    merged = load_yaml_with_inheritance(path)
    header = f"# merged from: {path.resolve().as_posix()}\n"
    body = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
    return header + body


# CLI用のエントリポイント
def main():
    import argparse
    parser = argparse.ArgumentParser(description="YAML設定からTrainConfigを生成")
    parser.add_argument("config", help="設定YAMLファイルのパス")
    parser.add_argument("--dry-run", action="store_true", help="設定を表示するだけで実行しない")
    args = parser.parse_args()
    
    exp_config = ExperimentConfig.from_yaml(args.config)
    print(exp_config.summary())
    
    if not args.dry_run:
        train_config = exp_config.to_train_config()
        print(f"\nGenerated TrainConfig: {train_config.name}")


if __name__ == "__main__":
    main()
