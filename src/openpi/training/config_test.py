import pathlib

import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config


def _apply_data_transforms(data_config: _config.DataConfig, sample: dict) -> dict:
    data = sample
    for transform in data_config.repack_transforms.inputs:
        data = transform(data)
    for transform in data_config.data_transforms.inputs:
        data = transform(data)
    return data


def _make_hsr_sample(action_dims: dict[str, int]) -> dict:
    sample = {
        "observation.image.head": np.zeros((64, 64, 3), dtype=np.uint8),
        "observation.image.hand": np.zeros((64, 64, 3), dtype=np.uint8),
        "observation.state": np.zeros((8,), dtype=np.float32),
        "prompt": "test",
    }
    for key, dim in action_dims.items():
        sample[f"action.{key}"] = np.zeros((2, dim), dtype=np.float32)
    return sample


def test_hsr_action_absolute_matches_relative_shape():
    model_config = pi0_config.Pi0Config(action_dim=16, action_horizon=2, max_token_len=8)
    assets_dir = pathlib.Path(".")

    relative_config = _config.LeRobotHSRDataConfig(action_mode="relative").create(assets_dir, model_config)
    absolute_config = _config.LeRobotHSRDataConfig(
        action_mode="absolute_arm_head_relative_gripper_base"
    ).create(assets_dir, model_config)

    relative = _apply_data_transforms(relative_config, _make_hsr_sample({"relative": 11}))
    absolute = _apply_data_transforms(
        absolute_config,
        _make_hsr_sample({"absolute": 8, "relative": 11}),
    )

    assert relative["actions"].shape == absolute["actions"].shape
    assert relative["actions"].shape == (2, 16)
