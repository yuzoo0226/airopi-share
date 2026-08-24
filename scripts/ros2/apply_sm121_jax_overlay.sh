#!/usr/bin/env bash
# =============================================================================
#  Make the training container's JAX usable on sm_121 (NVIDIA GB10 / DGX Spark).
#
#  uv.lock pins jax[cuda12]==0.5.3, whose XLA does not know sm_121 and aborts
#  with "LLVM ERROR: Unsupported rounding mode for conversion" the moment a
#  float16/bfloat16 kernel is compiled — which pi0/pi0.5 training does.
#  jaxlib 0.6.2 compiles f32/f16/bf16 correctly there.
#
#  Two uv quirks this script works around:
#    * `uv run` (without --no-sync) re-syncs the environment to uv.lock and
#      silently undoes this overlay, together with the GB200 torch overlay. Use
#      ${VENV}/bin/python or `uv run --no-sync` afterwards.
#    * `uv pip install` executed inside the project honours
#      [tool.uv] override-dependencies (ml-dtypes==0.4.1, tensorstore==0.1.74),
#      which is exactly what has to move for jax 0.6.2 — so the installs below
#      run from outside the project directory.
#
#  Usage (inside the training container):
#      bash /home/openpi/scripts/ros2/apply_sm121_jax_overlay.sh
#  or from the host:
#      docker exec <container> bash /home/openpi/scripts/ros2/apply_sm121_jax_overlay.sh
# =============================================================================
set -euo pipefail

VENV="${UV_PROJECT_ENVIRONMENT:-/home/cache/venv}"
JAX_VERSION="${OPENPI_SM121_JAX_VERSION:-0.6.2}"
ORBAX_VERSION="${OPENPI_SM121_ORBAX_VERSION:-0.11.14}"
PY="${VENV}/bin/python"

if [ ! -x "${PY}" ]; then
    echo "[ERROR] no interpreter at ${PY}; run the container init (uv sync) first." >&2
    exit 1
fi

echo "[INFO] venv           : ${VENV}"
echo "[INFO] jax            : ${JAX_VERSION}"
echo "[INFO] orbax          : ${ORBAX_VERSION}"

# Run from / so the project's uv overrides do not pin ml-dtypes/tensorstore.
cd /

VIRTUAL_ENV="${VENV}" uv pip install --python "${PY}" \
    "jax[cuda12]==${JAX_VERSION}" \
    "orbax-checkpoint==${ORBAX_VERSION}" \
    "ml-dtypes>=0.5.0" \
    "tensorstore>=0.1.75"

# jax 0.6.x accepts numpy 2, but openpi does not; pin it back last.
VIRTUAL_ENV="${VENV}" uv pip install --python "${PY}" "numpy>=1.22.4,<2"

"${PY}" - <<'PYEOF'
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy
import orbax.checkpoint as ocp

print(f"jax={jax.__version__} orbax={ocp.__version__} ml_dtypes={ml_dtypes.__version__} numpy={numpy.__version__}")
device = jax.devices()[0]
print(f"device={device} platform={device.platform} cc={getattr(device, 'compute_capability', '?')}")
if device.platform == "gpu":
    a = jnp.ones((256, 256), jnp.bfloat16)
    print("bfloat16 matmul:", float((a @ a).block_until_ready().mean()))
else:
    print("no GPU visible; skipped the bfloat16 check")
PYEOF

echo "[INFO] done. Use ${PY} (or 'uv run --no-sync') so that uv does not revert this."
