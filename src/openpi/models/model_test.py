from flax import nnx
import jax
import pytest

from openpi.models import finetune as _finetune
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


def test_pi05_recipe_vision_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        finetune_recipe=_finetune.FineTuneRecipeConfig(
            freeze_text_tower=True,
            train_action_expert=True,
            train_action_head=True,
            vision_lora=_finetune.VisionLoRAConfig(enabled=True),
        ),
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    model_state = nnx.state(model)
    image_lora_paths = [
        "/".join(str(part) for part in path)
        for path, _ in model_state.flat_state().items()
    ]
    lora_state_elems = [
        path
        for path in image_lora_paths
        if path.startswith("PaliGemma/img") and "lora" in path
    ]
    assert lora_state_elems


def test_pi05_per_image_recipe_vision_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        image_encoder_mode="per_image",
        finetune_recipe=_finetune.FineTuneRecipeConfig(
            freeze_text_tower=True,
            train_action_expert=True,
            train_action_head=True,
            vision_lora=_finetune.VisionLoRAConfig(enabled=True),
        ),
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    model_state = nnx.state(model)
    image_paths = [
        "/".join(str(part) for part in path)
        for path, _ in model_state.flat_state().items()
    ]
    assert any(path.startswith("PaliGemma/img_left_wrist_0_rgb/") for path in image_paths)
    assert any(path.startswith("PaliGemma/img_right_wrist_0_rgb/") for path in image_paths)
    assert any(path.startswith("PaliGemma/img_left_wrist_0_rgb/") and "lora" in path for path in image_paths)
    assert any(path.startswith("PaliGemma/img_right_wrist_0_rgb/") and "lora" in path for path in image_paths)


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
