import flax.nnx as nnx
import jax

import openpi.models.finetune as _finetune
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def _get_trainable_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(freeze_filter))).flat_state()


def _flat_paths(state: nnx.State) -> list[str]:
    return ["/".join(str(part) for part in path) for path in state]


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_dummy_uses_tiny_vision_encoder():
    config = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    flat_state = nnx.state(abstract_model).flat_state()

    embedding_kernel = flat_state[("PaliGemma", "img", "embedding", "kernel")]
    assert embedding_kernel.shape == (14, 14, 3, 32)


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_pi0_recipe_vision_lora_freezes_text_tower_and_vision_base_only():
    config = _pi0_config.Pi0Config(
        finetune_recipe=_finetune.FineTuneRecipeConfig(
            freeze_text_tower=True,
            train_action_expert=True,
            train_action_head=True,
            vision_lora=_finetune.VisionLoRAConfig(enabled=True),
        )
    )

    frozen_paths = _flat_paths(_get_frozen_state(config))
    trainable_paths = _flat_paths(_get_trainable_state(config))

    assert any(path.startswith("PaliGemma/llm/") and "_1" not in path for path in frozen_paths)
    assert any(path.startswith("PaliGemma/img/") and "lora" not in path for path in frozen_paths)
    assert not any(path.startswith("PaliGemma/llm/") and "_1" in path for path in frozen_paths)
    assert not any(path.startswith(("action_", "state_proj")) for path in frozen_paths)

    assert any(path.startswith("PaliGemma/img/") and "lora" in path for path in trainable_paths)
    assert any(path.startswith("PaliGemma/llm/") and "_1" in path for path in trainable_paths)
    assert any(path.startswith("action_in_proj/") for path in trainable_paths)
    assert any(path.startswith("action_out_proj/") for path in trainable_paths)
    assert any(path.startswith("state_proj/") for path in trainable_paths)
    assert any(path.startswith("action_time_mlp_in/") for path in trainable_paths)
    assert any(path.startswith("action_time_mlp_out/") for path in trainable_paths)


def test_pi0_recipe_per_image_vision_lora_tracks_split_image_encoders():
    config = _pi0_config.Pi0Config(
        image_encoder_mode="per_image",
        finetune_recipe=_finetune.FineTuneRecipeConfig(
            freeze_text_tower=True,
            train_action_expert=True,
            train_action_head=True,
            vision_lora=_finetune.VisionLoRAConfig(enabled=True),
        ),
    )

    frozen_paths = _flat_paths(_get_frozen_state(config))
    trainable_paths = _flat_paths(_get_trainable_state(config))

    assert any(path.startswith("PaliGemma/img_left_wrist_0_rgb/") for path in frozen_paths)
    assert any(path.startswith("PaliGemma/img_right_wrist_0_rgb/") for path in frozen_paths)
    assert any(path.startswith("PaliGemma/img_left_wrist_0_rgb/") and "lora" in path for path in trainable_paths)
    assert any(path.startswith("PaliGemma/img_right_wrist_0_rgb/") and "lora" in path for path in trainable_paths)


def test_pi0_recipe_frozen_vision_keeps_vision_lora_but_freezes_it():
    config = _pi0_config.Pi0Config(
        image_encoder_mode="per_image",
        finetune_recipe=_finetune.FineTuneRecipeConfig(
            freeze_text_tower=True,
            train_action_expert=True,
            train_action_head=True,
            vision_train_mode="frozen",
            vision_lora=_finetune.VisionLoRAConfig(enabled=True),
        ),
    )

    frozen_paths = _flat_paths(_get_frozen_state(config))
    trainable_paths = _flat_paths(_get_trainable_state(config))

    assert any(path.startswith("PaliGemma/img/") and "lora" in path for path in frozen_paths)
    assert any(path.startswith("PaliGemma/img_left_wrist_0_rgb/") and "lora" in path for path in frozen_paths)
    assert not any(path.startswith("PaliGemma/img") for path in trainable_paths)
    assert any(path.startswith("PaliGemma/llm/") and "_1" in path for path in trainable_paths)
    assert any(path.startswith("action_out_proj/") for path in trainable_paths)


def test_pi0_recipe_full_vision_training_leaves_vision_params_trainable():
    config = _pi0_config.Pi0Config(
        image_encoder_mode="per_image",
        finetune_recipe=_finetune.FineTuneRecipeConfig(
            freeze_text_tower=False,
            train_action_expert=True,
            train_action_head=True,
            vision_train_mode="full",
            vision_lora=_finetune.VisionLoRAConfig(enabled=True),
        ),
    )

    frozen_paths = _flat_paths(_get_frozen_state(config))
    trainable_paths = _flat_paths(_get_trainable_state(config))

    assert not any(path.startswith("PaliGemma/img") for path in frozen_paths)
    assert any(path.startswith("PaliGemma/img/") and "lora" not in path for path in trainable_paths)
    assert any(path.startswith("PaliGemma/img_left_wrist_0_rgb/") and "lora" in path for path in trainable_paths)
    assert any(path.startswith("PaliGemma/llm/") and "_1" not in path for path in trainable_paths)
