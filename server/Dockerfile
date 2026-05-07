FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# --- Stage 1: Install dependencies (cached unless pyproject.toml/uv.lock change) ---
# Copy only dependency manifests + minimal stubs so hatchling can resolve
# the editable install without needing the full source tree.
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY packages/openpi-client/pyproject.toml ./packages/openpi-client/pyproject.toml
RUN mkdir -p packages/openpi-client/src/openpi_client && touch packages/openpi-client/src/openpi_client/__init__.py
RUN mkdir -p src/openpi && touch src/openpi/__init__.py

RUN uv venv --python 3.11 /workspace/.venv
# The serving image does not construct LeRobot datasets, so we intentionally skip
# the training-only lerobot -> av dependency chain. av==14.4.0 is source-only in
# uv.lock and would otherwise require a full FFmpeg toolchain during image build.
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-package lerobot --no-install-package av
# pytest is a dev dependency in pyproject.toml / uv.lock, but a runtime import in
# openpi.models_pytorch.gemma_pytorch currently references it for type annotations.
RUN uv pip install --python /workspace/.venv/bin/python pytest==8.3.5

# --- Stage 2: Copy source code (changes here do NOT re-run uv sync) ---
COPY packages/openpi-client ./packages/openpi-client
COPY src/openpi ./src/openpi
COPY configs ./configs
COPY server ./server

# Support PyTorch checkpoints in the same way as the existing OpenPI serving flow.
RUN /workspace/.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /workspace/src/openpi/models_pytorch/transformers_replace/* {}

RUN chmod +x /workspace/server/entrypoint.sh

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/openpi-client/src"
ENV OPENPI_DATA_HOME="/policy_assets"

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
