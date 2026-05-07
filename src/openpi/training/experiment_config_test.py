import flax.nnx as nnx
import pytest

from openpi.training.experiment_config import ExperimentConfig
from openpi.training.experiment_config import load_experiment_config
import openpi.training.weight_loaders as weight_loaders


def test_load_pi05_hsr_fast_multinode_yaml() -> None:
    cfg = load_experiment_config("configs/experiments/pi05_hsr_68tasks_fast_multinode_8nodes.yaml")
    base = cfg.data.base_config

    assert cfg.name == "pi05_hsr_68tasks_fast_multinode_8nodes"
    assert cfg.data.repo_id == "lerobot_datasets/airoa-hsr-all-v1.0-202504-202512-68tasks"
    assert cfg.data.action_mode == "state_diff_arm_head_relative_gripper_base"

    assert base is not None
    assert base.fast_lerobot is True
    assert base.lerobot_backend == "parquet"
    assert base.lerobot_checks == "none"
    assert base.video_backend == "torchcodec"

    assert cfg.batch_size == 64
    assert cfg.num_workers == 8
    assert cfg.pin_memory is True
    assert cfg.prefetch_factor == 8
    assert cfg.fsdp_devices == 8
    assert cfg.val_split_fraction == 0.1
    assert cfg.val_split_seed == 42
    assert cfg.eval_interval == 1000
    assert cfg.eval_num_batches == 20
    assert cfg.eval_num_sample_steps == 10
    assert cfg.task_sampler.kind == "adaptive"


def test_num_workers_defaults_to_local_gpu_count(tmp_path, monkeypatch) -> None:
    yaml_path = tmp_path / "auto_num_workers.yaml"
    yaml_path.write_text(
        """
experiment:
  name: auto_num_workers
dataset:
  repo_id: fake
training:
  batch_size: 8
gpu:
  num_gpus: 64
  base_gpus: 64
""".strip()
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.num_workers == 8


def test_num_workers_uses_yaml_value_when_specified(tmp_path, monkeypatch) -> None:
    yaml_path = tmp_path / "yaml_num_workers.yaml"
    yaml_path.write_text(
        """
experiment:
  name: yaml_num_workers
dataset:
  repo_id: fake
training:
  batch_size: 8
  num_workers: 3
gpu:
  num_gpus: 64
  base_gpus: 64
""".strip()
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.num_workers == 3


def test_pi05_model_defaults_to_full_variants(tmp_path) -> None:
    yaml_path = tmp_path / "pi05_defaults.yaml"
    yaml_path.write_text(
        """
experiment:
  name: pi05_defaults
dataset:
  repo_id: fake
model:
  type: pi05
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.model.pi05 is True
    assert cfg.model.paligemma_variant == "gemma_2b"
    assert cfg.model.action_expert_variant == "gemma_300m"


def test_pi05_lora_variants_are_applied_from_yaml(tmp_path) -> None:
    yaml_path = tmp_path / "pi05_lora.yaml"
    yaml_path.write_text(
        """
experiment:
  name: pi05_lora
dataset:
  repo_id: fake
model:
  type: pi05
  variant: lora
  paligemma_variant: gemma_2b_lora
  action_expert_variant: gemma_300m
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.model.pi05 is True
    assert cfg.model.paligemma_variant == "gemma_2b_lora"
    assert cfg.model.action_expert_variant == "gemma_300m"
    assert not isinstance(cfg.freeze_filter, nnx.filterlib.Nothing)


def test_lora_variants_enable_freeze_without_variant_flag(tmp_path) -> None:
    yaml_path = tmp_path / "pi05_lora_without_variant_flag.yaml"
    yaml_path.write_text(
        """
experiment:
  name: pi05_lora_without_variant_flag
dataset:
  repo_id: fake
model:
  type: pi05
  paligemma_variant: gemma_2b_lora
  action_expert_variant: gemma_300m
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.model.pi05 is True
    assert cfg.model.paligemma_variant == "gemma_2b_lora"
    assert cfg.model.action_expert_variant == "gemma_300m"
    assert not isinstance(cfg.freeze_filter, nnx.filterlib.Nothing)


def test_freeze_action_expert_flag_is_not_required_for_lora_freeze(tmp_path) -> None:
    yaml_path = tmp_path / "pi05_lora_freeze_flag_false.yaml"
    yaml_path.write_text(
        """
experiment:
  name: pi05_lora_freeze_flag_false
dataset:
  repo_id: fake
model:
  type: pi05
  variant: lora
  freeze_action_expert: false
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.model.pi05 is True
    assert cfg.model.paligemma_variant == "gemma_2b_lora"
    assert cfg.model.action_expert_variant == "gemma_300m_lora"
    assert not isinstance(cfg.freeze_filter, nnx.filterlib.Nothing)


def test_pi05_finetune_recipe_enables_full_variants_and_recipe_freeze(tmp_path) -> None:
    yaml_path = tmp_path / "pi05_recipe.yaml"
    yaml_path.write_text(
        """
experiment:
  name: pi05_recipe
dataset:
  repo_id: fake
model:
  type: pi05
  finetune_recipe:
    freeze_text_tower: true
    train_action_expert: true
    train_action_head: true
    vision_lora:
      enabled: true
      rank: 8
      alpha: 8.0
      targets: [patch_embedding, attention, mlp, head]
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.model.pi05 is True
    assert cfg.model.paligemma_variant == "gemma_2b"
    assert cfg.model.action_expert_variant == "gemma_300m"
    assert cfg.model.finetune_recipe is not None
    assert cfg.model.finetune_recipe.freeze_text_tower is True
    assert cfg.model.finetune_recipe.vision_lora.enabled is True
    assert cfg.model.finetune_recipe.vision_train_mode == "lora"
    assert not isinstance(cfg.freeze_filter, nnx.filterlib.Nothing)


def test_finetune_recipe_rejects_legacy_lora_variants(tmp_path) -> None:
    yaml_path = tmp_path / "recipe_with_legacy_lora.yaml"
    yaml_path.write_text(
        """
experiment:
  name: recipe_with_legacy_lora
dataset:
  repo_id: fake
model:
  type: pi05
  paligemma_variant: gemma_2b_lora
  finetune_recipe:
    freeze_text_tower: true
    vision_lora:
      enabled: true
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    with pytest.raises(ValueError, match="LoRA model variants cannot be combined"):
        ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False)


def test_image_encoder_mode_can_be_set_from_yaml(tmp_path) -> None:
    yaml_path = tmp_path / "per_image_encoder.yaml"
    yaml_path.write_text(
        """
experiment:
  name: per_image_encoder
dataset:
  repo_id: fake
model:
  type: pi05
  image_encoder_mode: per_image
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.model.image_encoder_mode == "per_image"


def test_vision_train_mode_can_be_set_from_yaml(tmp_path) -> None:
    yaml_path = tmp_path / "frozen_vision_recipe.yaml"
    yaml_path.write_text(
        """
experiment:
  name: frozen_vision_recipe
dataset:
  repo_id: fake
model:
  type: pi05
  image_encoder_mode: per_image
  finetune_recipe:
    freeze_text_tower: true
    vision_train_mode: frozen
    vision_lora:
      enabled: true
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.model.finetune_recipe is not None
    assert cfg.model.finetune_recipe.vision_train_mode == "frozen"
    assert not isinstance(cfg.freeze_filter, nnx.filterlib.Nothing)


def test_vision_train_mode_lora_requires_vision_lora(tmp_path) -> None:
    yaml_path = tmp_path / "invalid_lora_vision_train_mode.yaml"
    yaml_path.write_text(
        """
experiment:
  name: invalid_lora_vision_train_mode
dataset:
  repo_id: fake
model:
  type: pi05
  finetune_recipe:
    vision_train_mode: lora
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
""".strip()
    )

    with pytest.raises(ValueError, match="vision_train_mode='lora'"):
        ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False)


def test_learning_rate_scaling_uses_batch_size_ratio_not_gpu_ratio(tmp_path) -> None:
    yaml_path = tmp_path / "batch_ratio_scaling.yaml"
    yaml_path.write_text(
        """
experiment:
  name: batch_ratio_scaling
dataset:
  repo_id: fake
training:
  batch_size: 1024
  num_train_steps: 5600000
  lr_schedule:
    peak_lr: 2.5e-5
    decay_lr: 2.5e-6
gpu:
  num_gpus: 64
  base_gpus: 1
scaling:
  scale_batch_size: false
  scale_learning_rate: true
  scale_train_steps: true
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert cfg.batch_size == 1024
    assert cfg.lr_schedule.peak_lr == pytest.approx(2.5e-5)
    assert cfg.lr_schedule.decay_lr == pytest.approx(2.5e-6)
    assert cfg.num_train_steps == 5600000


def test_learning_rate_and_steps_scale_from_base_batch_size(tmp_path) -> None:
    yaml_path = tmp_path / "base_batch_size_scaling.yaml"
    yaml_path.write_text(
        """
experiment:
  name: base_batch_size_scaling
dataset:
  repo_id: fake
training:
  batch_size: 1024
  num_train_steps: 5600000
  lr_schedule:
    peak_lr: 2.5e-5
    decay_lr: 2.5e-6
gpu:
  num_gpus: 64
  base_gpus: 1
scaling:
  base_batch_size: 128
  scale_batch_size: false
  scale_learning_rate: true
  scale_train_steps: true
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    expected_factor = 1024 / 128
    assert cfg.lr_schedule.peak_lr == pytest.approx(2.5e-5 * (expected_factor**0.5))
    assert cfg.lr_schedule.decay_lr == pytest.approx(2.5e-6 * (expected_factor**0.5))
    assert cfg.num_train_steps == int(5600000 / expected_factor)


def test_weight_loader_can_be_selected_as_paligemma_from_yaml(tmp_path) -> None:
    yaml_path = tmp_path / "paligemma_loader.yaml"
    yaml_path.write_text(
        """
experiment:
  name: paligemma_loader
dataset:
  repo_id: fake
model:
  type: pi0
training:
  batch_size: 8
gpu:
  num_gpus: 1
  base_gpus: 1
weight_loader:
  type: paligemma
""".strip()
    )

    cfg = ExperimentConfig.from_yaml(yaml_path, auto_detect_gpu=False).to_train_config()

    assert isinstance(cfg.weight_loader, weight_loaders.PaliGemmaWeightLoader)
