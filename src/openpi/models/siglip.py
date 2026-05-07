# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence
import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.finetune as _finetune
import openpi.models.lora as _lora
import openpi.training.sharding as sharding


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


def _vision_lora_config(
    vision_lora: _finetune.VisionLoRAConfig,
    target: _finetune.VisionLoRATarget,
) -> _lora.LoRAConfig | None:
    if not vision_lora.applies_to(target):
        return None
    return _lora.LoRAConfig(rank=vision_lora.rank, alpha=vision_lora.alpha)


class LoRADense(nn.Module):
    features: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    lora_config: _lora.LoRAConfig | None = None
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        y = jnp.dot(x, kernel.astype(x.dtype))
        if self.lora_config is not None:
            lora_a = self.param("lora_a", self.lora_config.init_fn, (x.shape[-1], self.lora_config.rank))
            lora_b = self.param("lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features))
            y = y + jnp.dot(jnp.dot(x, lora_a.astype(x.dtype)), lora_b.astype(x.dtype)) * self.lora_config.scaling_value
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            y = y + bias.astype(x.dtype)
        return y


class LoRAConv(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int]
    padding: str
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    lora_config: _lora.LoRAConfig | None = None
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (*self.kernel_size, x.shape[-1], self.features))
        if self.lora_config is not None:
            lora_a = self.param(
                "lora_a",
                self.lora_config.init_fn,
                (*self.kernel_size, x.shape[-1], self.lora_config.rank),
            )
            lora_b = self.param("lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features))
            kernel = kernel + jnp.einsum("hwcr,ro->hwco", lora_a, lora_b) * self.lora_config.scaling_value
        y = jax.lax.conv_general_dilated(
            x,
            kernel.astype(x.dtype),
            self.strides,
            self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            y = y + bias.astype(y.dtype)
        return y


class AttentionInputProjection(nn.Module):
    num_heads: int
    head_dim: int
    kernel_init: nn.initializers.Initializer
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    lora_config: _lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.num_heads, self.head_dim))
        y = jnp.einsum("...d,dnh->...nh", x, kernel.astype(x.dtype))
        if self.lora_config is not None:
            lora_a = self.param("lora_a", self.lora_config.init_fn, (x.shape[-1], self.lora_config.rank))
            lora_b = self.param(
                "lora_b",
                self.lora_config.init_fn,
                (self.lora_config.rank, self.num_heads, self.head_dim),
            )
            y = (
                y
                + jnp.einsum("...d,dr,rnh->...nh", x, lora_a.astype(x.dtype), lora_b.astype(x.dtype))
                * self.lora_config.scaling_value
            )
        bias = self.param("bias", self.bias_init, (self.num_heads, self.head_dim))
        return y + bias.astype(x.dtype)


class AttentionOutputProjection(nn.Module):
    features: int
    num_heads: int
    head_dim: int
    kernel_init: nn.initializers.Initializer
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    lora_config: _lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (self.num_heads, self.head_dim, self.features))
        y = jnp.einsum("...nh,nhd->...d", x, kernel.astype(x.dtype))
        if self.lora_config is not None:
            lora_a = self.param("lora_a", self.lora_config.init_fn, (self.num_heads, self.head_dim, self.lora_config.rank))
            lora_b = self.param("lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features))
            y = (
                y
                + jnp.einsum("...nh,nhr,rd->...d", x, lora_a.astype(x.dtype), lora_b.astype(x.dtype))
                * self.lora_config.scaling_value
            )
        bias = self.param("bias", self.bias_init, (self.features,))
        return y + bias.astype(x.dtype)


class MultiHeadDotProductAttention(nn.Module):
    num_heads: int = 12
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    dtype_mm: str = "float32"
    lora_config: _lora.LoRAConfig | None = None

    @nn.compact
    def __call__(self, inputs_q, inputs_kv):
        features = inputs_q.shape[-1]
        if features % self.num_heads != 0:
            raise ValueError(f"Embedding dim ({features}) must be divisible by num_heads ({self.num_heads}).")
        head_dim = features // self.num_heads
        dtype = inputs_q.dtype

        query = AttentionInputProjection(
            self.num_heads,
            head_dim,
            kernel_init=self.kernel_init,
            lora_config=self.lora_config,
            name="query",
        )(inputs_q)
        key = AttentionInputProjection(
            self.num_heads,
            head_dim,
            kernel_init=self.kernel_init,
            lora_config=self.lora_config,
            name="key",
        )(inputs_kv)
        value = AttentionInputProjection(
            self.num_heads,
            head_dim,
            kernel_init=self.kernel_init,
            lora_config=self.lora_config,
            name="value",
        )(inputs_kv)

        scale = head_dim**-0.5
        logits = jnp.einsum(
            "BTNH,BSNH->BNTS",
            query * scale,
            key,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        weights = jax.nn.softmax(logits, axis=-1).astype(dtype)
        attended = jnp.einsum("BNTS,BSNH->BTNH", weights, value)
        out = AttentionOutputProjection(
            features,
            self.num_heads,
            head_dim,
            kernel_init=self.kernel_init,
            lora_config=self.lora_config,
            name="out",
        )(attended)
        return out.astype(self.dtype_mm)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"
    vision_lora: _finetune.VisionLoRAConfig = dataclasses.field(default_factory=_finetune.VisionLoRAConfig)

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = LoRADense(
            self.mlp_dim or 4 * d,
            lora_config=_vision_lora_config(self.vision_lora, "mlp"),
            name="Dense_0",
            **inits,
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return LoRADense(
            d,
            lora_config=_vision_lora_config(self.vision_lora, "mlp"),
            name="Dense_1",
            **inits,
        )(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    vision_lora: _finetune.VisionLoRAConfig = dataclasses.field(default_factory=_finetune.VisionLoRAConfig)

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype_mm=self.dtype_mm,
            lora_config=_vision_lora_config(self.vision_lora, "attention"),
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            vision_lora=self.vision_lora,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    vision_lora: _finetune.VisionLoRAConfig = dataclasses.field(default_factory=_finetune.VisionLoRAConfig)

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                vision_lora=self.vision_lora,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    vision_lora=self.vision_lora,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"
    vision_lora: _finetune.VisionLoRAConfig = dataclasses.field(default_factory=_finetune.VisionLoRAConfig)

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype_mm=self.dtype_mm,
            lora_config=_vision_lora_config(self.vision_lora, "attention"),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype_mm=self.dtype_mm, vision_lora=self.vision_lora)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    vision_lora: _finetune.VisionLoRAConfig = dataclasses.field(default_factory=_finetune.VisionLoRAConfig)

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction
        x = out["stem"] = LoRAConv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            lora_config=_vision_lora_config(self.vision_lora, "patch_embedding"),
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            vision_lora=self.vision_lora,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype_mm=self.dtype_mm,
                vision_lora=self.vision_lora,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = LoRADense(
                self.num_classes,
                bias_init=nn.initializers.zeros_init(),
                lora_config=_vision_lora_config(self.vision_lora, "head"),
                name="head",
                **kw,
            )
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
