#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-/home/openpi}"
VENV_DIR="${UV_PROJECT_ENVIRONMENT:-${REPO_ROOT}/.venv}"
VENV_PYTHON="${VENV_DIR}/bin/python"
ENV_FILE="${VENV_DIR}/.gb200_overlay_env.sh"

TORCH_INDEX_URL="${OPENPI_GB200_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${OPENPI_GB200_TORCH_VERSION:-2.10.0+cu128}"
TORCHVISION_VERSION="${OPENPI_GB200_TORCHVISION_VERSION:-0.25.0+cu128}"
TORCHCODEC_VERSION="${OPENPI_GB200_TORCHCODEC_VERSION:-0.10.0+cu128}"
TORCH_CUDA_VERSION="${OPENPI_GB200_TORCH_CUDA_VERSION:-12.8}"
STAMP_FILE="${VENV_DIR}/.gb200_overlay_stamp"
STAMP_CONTENT="${TORCH_INDEX_URL}|${TORCH_VERSION}|${TORCHVISION_VERSION}|${TORCHCODEC_VERSION}|${TORCH_CUDA_VERSION}|$(uname -m)"

if [ ! -x "${VENV_PYTHON}" ]; then
  echo "[GB200 overlay] Python venv not found: ${VENV_PYTHON}" >&2
  echo "[GB200 overlay] Run 'uv sync' before applying the overlay." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[GB200 overlay] uv is required but not available in PATH." >&2
  exit 1
fi

overlay_matches() {
  if [ ! -f "${STAMP_FILE}" ]; then
    return 1
  fi
  if [ "$(cat "${STAMP_FILE}")" != "${STAMP_CONTENT}" ]; then
    return 1
  fi

  OPENPI_EXPECTED_TORCH_VERSION="${TORCH_VERSION}" \
    OPENPI_EXPECTED_TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
    OPENPI_EXPECTED_TORCHCODEC_VERSION="${TORCHCODEC_VERSION}" \
    OPENPI_EXPECTED_TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION}" \
    "${VENV_PYTHON}" - <<'PY' >/dev/null 2>&1
import os
from importlib import metadata

import torch

assert metadata.version("torch") == os.environ["OPENPI_EXPECTED_TORCH_VERSION"]
assert metadata.version("torchvision") == os.environ["OPENPI_EXPECTED_TORCHVISION_VERSION"]
assert metadata.version("torchcodec") == os.environ["OPENPI_EXPECTED_TORCHCODEC_VERSION"]
assert torch.version.cuda == os.environ["OPENPI_EXPECTED_TORCH_CUDA_VERSION"]
PY
}

if overlay_matches; then
  echo "[GB200 overlay] Existing torch overlay matches requested versions. Skipping."
else
  echo "[GB200 overlay] Installing torch ${TORCH_VERSION}, torchvision ${TORCHVISION_VERSION}, torchcodec ${TORCHCODEC_VERSION}"
  uv pip install --python "${VENV_PYTHON}" --no-cache-dir --upgrade \
    --index "${TORCH_INDEX_URL}" \
    --default-index https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchcodec==${TORCHCODEC_VERSION}"
  printf '%s\n' "${STAMP_CONTENT}" > "${STAMP_FILE}"
fi

VENV_SITE_PACKAGES="$("${VENV_PYTHON}" - <<'PY'
import site

for path in site.getsitepackages():
    if "site-packages" in path:
        print(path)
        break
else:
    raise SystemExit("site-packages not found")
PY
)"

TORCH_LIB_DIR="${VENV_SITE_PACKAGES}/torch/lib"
GB200_LD_LIBRARY_PATH="${TORCH_LIB_DIR}"
if [ -d "${VENV_SITE_PACKAGES}/nvidia" ]; then
  while IFS= read -r lib_dir; do
    GB200_LD_LIBRARY_PATH="${GB200_LD_LIBRARY_PATH}:${lib_dir}"
  done < <(find "${VENV_SITE_PACKAGES}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
fi

cat > "${ENV_FILE}" <<EOF
#!/bin/bash
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${GB200_LD_LIBRARY_PATH}:\${LD_LIBRARY_PATH:-}"
EOF
chmod +x "${ENV_FILE}"

"${VENV_PYTHON}" - <<'PY'
from importlib import metadata

import torch

print(f"[GB200 overlay] torch={metadata.version('torch')} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"[GB200 overlay] torchvision={metadata.version('torchvision')}")
print(f"[GB200 overlay] torchcodec={metadata.version('torchcodec')}")
PY

if command -v ffmpeg >/dev/null 2>&1; then
  echo "[GB200 overlay] ffmpeg=$(ffmpeg -version | head -n 1)"
fi
