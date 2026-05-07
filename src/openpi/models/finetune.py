import dataclasses
from typing import Any, Literal

VisionLoRATarget = Literal["patch_embedding", "attention", "mlp", "head"]
VisionTrainMode = Literal["full", "lora", "frozen"]

_VISION_LORA_TARGETS = ("patch_embedding", "attention", "mlp", "head")
_VISION_TRAIN_MODES = ("full", "lora", "frozen")


@dataclasses.dataclass(frozen=True)
class VisionLoRAConfig:
    enabled: bool = False
    rank: int = 16
    alpha: float = 16.0
    targets: tuple[VisionLoRATarget, ...] = _VISION_LORA_TARGETS

    def __post_init__(self) -> None:
        normalized_targets = tuple(str(target).strip().lower() for target in self.targets)
        invalid_targets = tuple(target for target in normalized_targets if target not in _VISION_LORA_TARGETS)
        if invalid_targets:
            raise ValueError(
                f"Unsupported vision_lora.targets: {invalid_targets!r}. "
                f"Supported: {', '.join(_VISION_LORA_TARGETS)}."
            )
        if self.rank <= 0:
            raise ValueError(f"vision_lora.rank must be > 0, got {self.rank}.")
        if self.alpha <= 0:
            raise ValueError(f"vision_lora.alpha must be > 0, got {self.alpha}.")
        if self.enabled and not normalized_targets:
            raise ValueError("vision_lora.targets must not be empty when vision_lora.enabled is true.")
        object.__setattr__(self, "targets", normalized_targets)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VisionLoRAConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("model.finetune_recipe.vision_lora must be a mapping.")
        targets = data.get("targets", _VISION_LORA_TARGETS)
        if isinstance(targets, list):
            targets = tuple(targets)
        if not isinstance(targets, tuple):
            raise ValueError("model.finetune_recipe.vision_lora.targets must be a sequence.")
        return cls(
            enabled=bool(data.get("enabled", False)),
            rank=int(data.get("rank", 16)),
            alpha=float(data.get("alpha", 16.0)),
            targets=targets,
        )

    def applies_to(self, target: VisionLoRATarget) -> bool:
        return self.enabled and target in self.targets


@dataclasses.dataclass(frozen=True)
class FineTuneRecipeConfig:
    freeze_text_tower: bool = False
    train_action_expert: bool = True
    train_action_head: bool = True
    vision_lora: VisionLoRAConfig = dataclasses.field(default_factory=VisionLoRAConfig)
    vision_train_mode: VisionTrainMode | None = None

    def __post_init__(self) -> None:
        mode = self.vision_train_mode
        if mode is None:
            mode = "lora" if self.vision_lora.enabled else "full"
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in _VISION_TRAIN_MODES:
            raise ValueError(
                f"Unsupported vision_train_mode: {mode!r}. "
                f"Supported: {', '.join(_VISION_TRAIN_MODES)}."
            )
        if normalized_mode == "lora" and not self.vision_lora.enabled:
            raise ValueError("vision_train_mode='lora' requires model.finetune_recipe.vision_lora.enabled=true.")
        object.__setattr__(self, "vision_train_mode", normalized_mode)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FineTuneRecipeConfig | None":
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("model.finetune_recipe must be a mapping.")
        return cls(
            freeze_text_tower=bool(data.get("freeze_text_tower", False)),
            train_action_expert=bool(data.get("train_action_expert", True)),
            train_action_head=bool(data.get("train_action_head", True)),
            vision_lora=VisionLoRAConfig.from_dict(data.get("vision_lora")),
            vision_train_mode=data.get("vision_train_mode"),
        )

    @property
    def has_overrides(self) -> bool:
        return (
            self.freeze_text_tower
            or not self.train_action_expert
            or not self.train_action_head
            or self.vision_lora.enabled
            or self.vision_train_mode != "full"
        )
