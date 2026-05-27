# AiroPi for HSR — Unified Guide

Fine-tuning, evaluation, and deployment of π0 / π0.5 models for the HSR robot.
One codebase supports two runtime families:

- **Local GPU machine (Docker)** — reproducible training (non-Blackwell /
  Blackwell GB200 / HSR deploy images)
- **HPC (SLURM / PBS + qsub)** — distributed multi-node training

The stack is **pixi + uv + Docker** in layers, and local training is unified on
the Docker path. Running `pixi run uv run scripts/train.py ...` directly on a
local host ("pixi + uv standalone") is **not supported** at the moment; pixi is
invoked automatically from within the HPC launchers.

Experiment configs are **YAML-first** (`configs/experiments/*.yaml`). The legacy
tyro `CONFIG_NAME` path is preserved for single-node development, but the
multi-node launchers, Docker images, Blackwell overlay, and policy server are
all designed around YAML.

> 日本語版: See [`README_ja.md`](README_ja.md) for the Japanese comprehensive guide.
> PyTorch training is supported but shows worse performance than the original
> JAX implementation on this codebase — prefer JAX for HSR training.

---

## Table of contents

1. [Overview & compatibility matrix](#1-overview--compatibility-matrix)
2. [Repository layout](#2-repository-layout)
3. [Quick start (TL;DR)](#3-quick-start-tldr)
4. [Prerequisites](#4-prerequisites)
5. [Experiment configuration (YAML-first)](#5-experiment-configuration-yaml-first)
6. [Local environment](#6-local-environment)
7. [Model deployment](#7-model-deployment)
8. [HPC — SLURM](#8-hpc--slurm)
9. [HPC — PBS (qsub)](#9-hpc--pbs-qsub)
10. [Cross-cutting tips](#10-cross-cutting-tips)
11. [.env / .env.local cheatsheet](#11-env--envlocal-cheatsheet)
12. [Troubleshooting](#12-troubleshooting)
13. [Appendix](#13-appendix)

---

## 1. Overview & compatibility matrix

| Use case | Recommended script(s) | Config style | Notes |
| --- | --- | --- | --- |
| Local Docker (non-Blackwell) | `BUILD-DOCKER-CONTAINER.sh train` + `RUN-DOCKER-CONTAINER.sh train` | YAML | CUDA 12.2 base, **standard local path** |
| Local Docker (Blackwell GB200) | `BUILD-DOCKER-CONTAINER.sh train-gb200` + `RUN-DOCKER-CONTAINER.sh train-gb200` | YAML | Overlays cu128 torch |
| Local Docker (HSR deploy) | `BUILD-DOCKER-CONTAINER.sh deploy` + `RUN-DOCKER-CONTAINER.sh deploy` | — | ROS Noetic included |
| Policy server image | `docker build -f server/Dockerfile -t openpi-serve .` | YAML / tyro | WebSocket-based remote inference |
| HPC SLURM single-node | `RUN-UV-SBATCH.sh` / `RUN-UV-SBATCH-8GPU.sh` | tyro (`CONFIG_NAME`) | Script invokes pixi+uv internally |
| HPC SLURM multi-node | `RUN-UV-SBATCH-OPENPI-MULTINODE.sh` | YAML | Assumes 8 GPU × N nodes |
| HPC PBS single-node | `RUN-UV.sh` | YAML / tyro | Script invokes pixi+uv internally |
| HPC PBS multi-node (preferred) | `RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh` | YAML | HPCX + direct mpirun |
| HPC PBS multi-node (full-featured) | `RUN-UV-QSUB-OPENPI-MULTINODE.sh` | YAML | UCX tuning, preflight |
| HPC Slurm interactive allocation | `./run_interactive_node.sh` | — | `salloc` wrapper; use `sbatch`-based launcher after shell |
| HPC Singularity single-node | `BUILD_SINGULARITY.sh` + `RUN-SINGULARITY*.sh` | tyro / YAML | For Docker-less sites |

> **Running `pixi run uv run scripts/train.py ...` directly on the local host is
> not currently supported.** Always use Docker (`train` / `train-gb200`) for
> local training.

---

## 2. Repository layout

```
AiroPi/
├── README.md                        # English comprehensive guide
├── README_ja.md                     # Japanese comprehensive guide
│
├── pixi.toml, pixi.lock             # System deps (ffmpeg 7 / pkg-config / compilers)
├── pyproject.toml, uv.lock          # Python deps (Python 3.11)
│
├── .env.example                     # Experiment settings template  (tracked)
├── .env.local.example               # Personal settings template    (tracked)
│                                    # Actual .env / .env.local are gitignored
│
├── BUILD-DOCKER-CONTAINER.sh        # docker compose build
├── RUN-DOCKER-CONTAINER.sh          # docker compose up + exec bash
├── RUN-DOCKER-COMMON.sh             # Shared logic for train / train-gb200 / deploy
├── RUN-DOCKER-TRAIN.sh              # In-container training helper
│
├── RUN-UV-COMMON.sh                 # Shared logic for pixi + uv launchers
├── RUN-UV.sh                        # PBS single-node (YAML / tyro)
├── RUN-UV-SBATCH.sh                 # SLURM single-node 1 GPU (tyro)
├── RUN-UV-SBATCH-8GPU.sh            # SLURM single-node 8 GPU (tyro)
├── RUN-UV-SBATCH-OPENPI-MULTINODE.sh     # SLURM multi-node (YAML)
├── RUN-UV-QSUB-OPENPI-MULTINODE.sh       # PBS multi-node (UCX-tuned)
├── RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh# PBS multi-node (SIMPLE, preferred)
├── RUN-UV-EVAL.sh, RUN-UV-DATA-CHECK.sh, ...
├── BUILD_SINGULARITY.sh, RUN-SINGULARITY*.sh  # Singularity path
├── run_interactive_node.sh          # Slurm interactive allocation (salloc)
│
├── docker/
│   ├── docker-compose.yml           # Mode-agnostic compose
│   ├── Dockerfile.train             # Non-Blackwell training (CUDA 12.2)
│   ├── Dockerfile.train.gb200       # Blackwell (cu122 + cu128 overlay)
│   ├── Dockerfile.deploy            # HSR deploy (ROS Noetic)
│   ├── openpi.def                   # Singularity definition
│   └── scripts/                     # init, GB200 overlay, ROS setup, …
│
├── server/
│   ├── Dockerfile                   # Policy server (lean runtime image)
│   ├── entrypoint.sh
│   └── serve_hsr_policy_ws.py       # WebSocket policy server
│
├── scripts/
│   ├── train.py                     # Training entrypoint
│   ├── compute_norm_stats.py        # tyro-based norm stats helper
│   ├── aggregate_stats_simple.py / aggregate_stats_fast.py
│   ├── eval_val_loss.py
│   ├── serve_policy.py              # Alternate server entrypoint (reference)
│   └── openpi_utils/                # SLURM / PBS distribution glue
│
├── configs/experiments/             # Experiment YAML
│   ├── example_base.yaml            # tracked template
│   ├── example_lora.yaml            # tracked template
│   ├── example_multinode.yaml       # tracked template
│   └── (other *.yaml are gitignored, local-only)
│
├── src/openpi/                      # Main source
├── packages/openpi-client/          # Client SDK (workspace subpackage)
└── deploy/                          # HSR ROS deployment assets
```

> `.gitignore` tracks **only** `configs/experiments/example_*.yaml`. Any other
> YAML you add under that directory is treated as local-only. To share a new
> experiment config, copy it from an `example_*.yaml`, rename it with the
> `example_` prefix, or update `.gitignore`.

---

## 3. Quick start (TL;DR)

### 3.1 Local Docker (non-Blackwell)

```bash
# On the host
export DEEP_PROJECT_NAME=mytest
export DEEP_DATASET_PATH=/path/to/datasets

./BUILD-DOCKER-CONTAINER.sh train
./RUN-DOCKER-CONTAINER.sh train
# → Docker exec'd you into the container; the rest runs inside /home/openpi:

# 1) Compute norm_stats for the YAML's dataset (required before training;
#    without it, the loader raises
#    "Normalization stats not found. Make sure to run scripts/compute_norm_stats.py ...")
JAX_PLATFORMS=cpu uv run python scripts/aggregate_stats_fast.py \
  --episodes-stats "${DATA_DIR}/meta/episodes_stats.jsonl" \
  --output-file "assets/example_base/${REPO_ID}/norm_stats.json" \
  --chunk-dir "${DATA_DIR}/data/" \
  --action-column "action.relative" \
  --action-mode "relative"

# 2) Train (YAML mode)
uv run scripts/train.py \
  --config-yaml configs/experiments/example_base.yaml \
  --exp-name my_exp --overwrite
```

- Point `DATA_DIR` / `REPO_ID` at whatever your YAML's `dataset.data_dir` /
  `dataset.repo_id` resolves to inside the container (typically under
  `/home/datasets/...`)
- The output path `assets/<assets_dir>/<asset_id>/norm_stats.json` comes from
  the YAML — `configs/experiments/example_base.yaml` uses `assets_dir:
  ./assets/example_base` and `asset_id: <repo_id>`. Your own YAML may rename
  both; keep the `aggregate_stats_fast.py --output-file` aligned.
- The first `uv sync` and cache setup run automatically from the container's
  init script — you do **not** need to run `pixi run sync` on the host.

### 3.2 Local Docker (Blackwell / GB200)

```bash
export DEEP_PROJECT_NAME=gb200test
export DEEP_DATASET_PATH=/path/to/datasets

./BUILD-DOCKER-CONTAINER.sh train-gb200
./RUN-DOCKER-CONTAINER.sh train-gb200
# The cu128 torch overlay is applied automatically inside the container.

# norm_stats → train (same as §3.1)
JAX_PLATFORMS=cpu uv run python scripts/aggregate_stats_fast.py \
  --episodes-stats "${DATA_DIR}/meta/episodes_stats.jsonl" \
  --output-file "assets/example_base/${REPO_ID}/norm_stats.json" \
  --chunk-dir "${DATA_DIR}/data/" \
  --action-column "action.relative" \
  --action-mode "relative"

uv run scripts/train.py \
  --config-yaml configs/experiments/example_base.yaml \
  --exp-name my_exp --overwrite
```

### 3.3 HPC PBS multi-node (preferred: SIMPLE)

```bash
cp .env.example .env              # experiment settings (YAML path, EXP_NAME, ...)
cp .env.local.example .env.local  # personal settings (WANDB_API_KEY, CACHE_ROOT, ...)
# Edit both files

qsub RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh
```

---

## 4. Prerequisites

### 4.1 Clone with submodules

This repo depends on LeRobot (submodule) and Git LFS. Use a recursive clone for
the first time:

```bash
git clone --recurse-submodules git@github.com:airoa-org/AiroPi.git
cd AiroPi
# If already cloned:
git submodule update --init --recursive
```

### 4.2 Copy `.env` and `.env.local`

Configuration is split across **two layers**:

- `.env` — experiment / launcher settings (YAML path, `EXP_NAME`, node counts,
  resume/overwrite, etc.). Intended to be team-shareable.
- `.env.local` — personal / machine-specific (WANDB key, `CACHE_ROOT`,
  `CHECKPOINT_DIR`, `HF_LEROBOT_HOME`, etc.). Intended to be per-developer.

Both templates (`.env.example`, `.env.local.example`) are committed. The actual
`.env` and `.env.local` are in `.gitignore`.

```bash
cp .env.example .env
cp .env.local.example .env.local
# Edit both files
```

**Load order** depends on the launcher family:

| Launcher family | Behavior | Notes |
| --- | --- | --- |
| **Family A**: everything using `RUN-UV-COMMON.sh::openpi_load_env_file`, plus `RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh` | Loads `.env` first, then `.env.local` (local wins) | `HSR_OPENPI_ENV_FILE` overrides the `.env` path |
| **Family B**: `RUN-SINGULARITY.sh`, `RUN-SINGULARITY-COMPUTE-SIMPLE-NORM.sh`, `download_base_model.sh`, `download_training_weights.sh` | Loads **only one** of `HSR_OPENPI_ENV_FILE` > `.env.local` > `.env` | Different semantics from Family A |
| **Family B exception**: `RUN-SINGULARITY-FT.sh` | Same as Family B, but override variable is `AIROPI_ENV_FILE` | Inconsistent with other Singularity scripts |

Minimum variables (details in §11):

- `.env`: `OPENPI_CONFIG_YAML`, `EXP_NAME`, `OPENPI_NUM_NODES`, `OPENPI_GRES`,
  `OPENPI_RESUME` / `OPENPI_OVERWRITE`
- `.env.local`: `WANDB_API_KEY`, `CACHE_ROOT`, `CHECKPOINT_DIR`, `HF_LEROBOT_HOME`

### 4.3 Where pixi + uv fit in

The stack is **pixi + uv + Docker**, layered as follows:

- **Docker (standard local path)**: bare `uv` is used inside the container. ffmpeg
  7 and compilers are baked in, so pixi is not needed
- **HPC (SLURM / PBS)**: the `RUN-UV-*.sh` launchers invoke `pixi run uv sync`
  and `pixi run uv run ...` on the login / compute nodes, so pixi and uv must
  be installed there
- **Host-side training**: not currently supported. Use Docker for local training

Install pixi and uv only if you manage the HPC cluster environment yourself
(or want a host-side venv for IDE / LSP reasons):

- [pixi installation](https://pixi.sh/latest/#installation)
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

> `pixi run sync` / `pixi run sync-blackwell` are still valid tasks in
> `pixi.toml` — run them if you want a local venv for source browsing. The
> official training path is the Docker flow in §6.

### 4.4 Datasets and base models

- Datasets follow the LeRobot layout (`${DATA_DIR}/<repo_id>/{data,meta,videos,...}`)
- `meta/episodes_stats.jsonl` is required for norm-stat computation
- Base models come from `checkpoints.base_model.<model_type>` in the YAML
  (e.g. `gs://openpi-assets/checkpoints/pi05_base/params`)
- For offline / multi-node setups, pre-download base models to
  `OPENPI_DATA_HOME` to avoid concurrent download races (see the
  `Pre-download base model weights` block in
  `RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh`). `download_base_model.sh` and
  `download_training_weights.sh` help with this.

---

## 5. Experiment configuration (YAML-first)

### 5.1 Responsibility split

```
.env                  Experiment / launcher settings
                      (OPENPI_CONFIG_YAML, EXP_NAME, node counts, resume/overwrite, preflight)
.env.local            Personal / credentials
                      (WANDB_API_KEY, CACHE_ROOT, CHECKPOINT_DIR, HF_LEROBOT_HOME)
configs/experiments/  Experiment semantics
*.yaml                (dataset / model / batch / LR / sampler / GPU count / base checkpoint)
```

The YAML captures the reproducible part of each run; `.env*` absorb machine and
user specifics.

### 5.2 YAML sections

| Section | Main fields |
| --- | --- |
| `experiment` | `name`, `project_name`, `seed`, `wandb_enabled` |
| `dataset` | `repo_id`, `data_dir`, `assets_dir`, `asset_id`, `action_mode`, `prompt_from_task`, `fast_lerobot`, `lerobot_backend`, `video_backend`, `adapt_to_pi`, `convert_gripper`, ... |
| `model` | `type` (`pi0` / `pi05`), `paligemma_variant`, `action_expert_variant`, `action_dim`, `action_horizon`, `finetune_recipe` (`vision_lora`, ...) |
| `training` | `batch_size`, `num_train_steps`, `pin_memory`, `prefetch_factor`, `eval_interval`, `save_interval`, `keep_period`, `lr_schedule` (e.g. `cosine_decay`) |
| `gpu` | `num_gpus`, `base_gpus`, `fsdp_devices` |
| `scaling` | `base_batch_size`, `scale_batch_size`, `scale_learning_rate`, `scale_train_steps`, `workers_per_gpu` |
| `task_sampler` | `kind` (`uniform` etc.), `alpha`, `ema_decay`, `min_prob` |
| `weight_loader` | `type` (`checkpoint` / `paligemma` / `noop`), `params_path` |
| `checkpoints` | `base_model.<type>` — base checkpoint URL per model type |

The full schema lives in `src/openpi/training/experiment_config.py`
(`ExperimentConfig.from_yaml`).

### 5.3 `_base_` inheritance

Start a YAML with `_base_: <filename>` to merge from another YAML in the same
directory (`load_yaml_with_inheritance` resolves recursively).

```yaml
# configs/experiments/example_lora.yaml (conceptual sketch)
_base_: example_base.yaml

experiment:
  name: my_lora_run

model:
  finetune_recipe:
    vision_train_mode: full
    vision_lora:
      enabled: true
      rank: 16
```

### 5.4 Tracked YAML templates

Only three YAMLs are committed:

- `configs/experiments/example_base.yaml` — minimal pi0.5 fine-tune
- `configs/experiments/example_lora.yaml` — LoRA example
- `configs/experiments/example_multinode.yaml` — multi-node GPU / scaling example

Other YAMLs are gitignored (`configs/experiments/*.yaml`, negated by
`!configs/experiments/example_*.yaml`). When you want to share a new config,
copy from one of the templates and use an `example_` prefix (or update the
gitignore rules).

### 5.5 Legacy tyro (`CONFIG_NAME`) path

When `--config-yaml` is not given, `scripts/train.py` treats the first
positional argument as a registered tyro `TrainConfig` name (defined in
`src/openpi/training/config.py`). Examples: `pi05_hsr`, `pi0_hsr`,
`pi0_aloha`, `pi05_aloha`, `pi0_libero`, plus many fine-tune variants.

```bash
# Inside the local Docker container:
uv run scripts/train.py pi05_hsr --exp-name my_exp --overwrite
# On HPC single-node SLURM (§8.1), the launcher uses tyro:
sbatch RUN-UV-SBATCH.sh
```

> The tyro path is handy for single-node dev, but **multi-node launchers,
> Blackwell overlay, and the policy server assume YAML**. Prefer YAML for new
> experiments.

---

## 6. Local environment

Local training on a GPU machine goes **through Docker only**. Three modes are
available (`train` / `train-gb200` / `deploy`). On the host you only need the
preparation from §4 (copy `.env` / `.env.local`) plus Docker and the NVIDIA
Container Toolkit.

> **Running `scripts/train.py` directly on the host via pixi + uv is not
> currently supported.** `pixi run sync` is only useful if you want a local venv
> for source browsing / IDE — not for training.

### 6.1 Docker: non-Blackwell training (`train`)

Builds from `docker/Dockerfile.train` (CUDA 12.2 / Ubuntu 22.04). In the Docker
path you do **not** use pixi — you use bare `uv` inside the container (ffmpeg 7
is already built into the image).

**Prerequisites**

- Docker + NVIDIA Container Toolkit (see
  `scripts/docker/install_docker_ubuntu22.sh` and
  `scripts/docker/install_nvidia_container_toolkit.sh`)
- Environment variables:
  - `DEEP_PROJECT_NAME` (required): compose project name. The container name
    becomes `${DEEP_PROJECT_NAME}_deep_1`
  - `DEEP_DATASET_PATH` (recommended): host path for datasets, mounted into
    `/home/datasets` inside the container
  - `DEEP_CACHE_ROOT_HOST` (optional): host cache directory for uv / HF / tmp;
    defaults to `./.docker_cache`

**Build**

```bash
export DEEP_PROJECT_NAME=review
export DEEP_DATASET_PATH=/path/to/lerobot_datasets
./BUILD-DOCKER-CONTAINER.sh train
```

**Run**

```bash
./RUN-DOCKER-CONTAINER.sh train
# docker compose up -d, then docker exec into the container
```

Once the container starts, `docker/scripts/initialize-docker-container.sh`
performs:

1. Create `/home/cache/*` (uv, HF, tmp, XDG)
2. Seed `.bashrc` (append `${UV_PROJECT_ENVIRONMENT}/bin` and source the GB200
   overlay env file if present; inside the container,
   `UV_PROJECT_ENVIRONMENT` defaults to `/home/cache/venv`, which maps to the
   host directory `.docker_cache/venv` — it is intentionally isolated from any
   host-side `.venv/`)
3. Initial `uv sync` (populates the venv outside of image layers)
4. Apply the GB200 overlay (only in `train-gb200` mode)
5. `tail -f /dev/null` to keep the container alive

Inside the container you land in `/home/openpi`. From there:

```bash
uv run scripts/train.py --config-yaml configs/experiments/example_base.yaml --exp-name my_exp --overwrite
```

Norm-stats computation works the same way (`uv run ...`).

**Volumes (`docker/docker-compose.yml`)**

| Host | Container | Purpose |
| --- | --- | --- |
| Repo root | `/home/openpi` | Source (writable) |
| `./docker/scripts` | `/home/docker_scripts` | init / overlay scripts |
| `${DEEP_CACHE_ROOT_HOST}` | `/home/cache` | uv / HF / tmp |
| `${DEEP_DATASET_PATH}` | `/home/datasets` | LeRobot datasets |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | X11 (GUI debug) |
| `/etc/passwd`, `/etc/group` (ro) | same | User resolution |
| `/dev/` | `/dev/` | GPU / USB |

Main CLI flags (`uv run scripts/train.py` inside the container):

| Flag | Meaning |
| --- | --- |
| `--config-yaml PATH` | YAML path (recommended, exclusive with tyro) |
| `--exp-name NAME` | Run name (overrides YAML `experiment.name`) |
| `--resume` | Resume from the latest checkpoint |
| `--overwrite` | Clear and re-create the checkpoint directory |
| `--checkpoint-base-dir PATH` | Override checkpoint base dir (default: `CHECKPOINT_DIR`) |
| `--assets-base-dir PATH` | Override the assets base dir |
| `--seed N` | Override the random seed |

Norm stats are computed inside the container:

```bash
JAX_PLATFORMS=cpu uv run python scripts/aggregate_stats_fast.py \
  --episodes-stats "${DATA_DIR}/meta/episodes_stats.jsonl" \
  --output-file "assets/<exp_name>/<repo_id>/norm_stats.json" \
  --chunk-dir "${DATA_DIR}/data/" \
  --action-column "action.relative" \
  --action-mode "relative"
```

Keep `--action-mode` aligned with YAML `dataset.action_mode` (current default:
`relative`).

### 6.2 Docker: Blackwell (GB200) training (`train-gb200`)

For GB200 (cu128 + aarch64) use `Dockerfile.train.gb200`. The
`docker/scripts/apply-gb200-overlay.sh` script runs automatically inside the
container to overlay cu128 torch.

**Outline**

- Base image: `nvidia/cuda:12.2.2-devel-ubuntu22.04`
- First-time `uv sync` installs cu122 torch (2.7.1) and jax (0.5.3)
- Then `docker/scripts/apply-gb200-overlay.sh` runs and:
  - Reinstalls `torch==2.10.0+cu128` / `torchvision==0.25.0+cu128` /
    `torchcodec==0.10.0+cu128` from the official cu128 index
  - Writes `${UV_PROJECT_ENVIRONMENT}/.gb200_overlay_env.sh` (adds cu128 torch
    libs and nvidia libs to `LD_LIBRARY_PATH`)
  - Sources that file from `.bashrc` **for interactive shells only**
  - Idempotence via `${UV_PROJECT_ENVIRONMENT}/.gb200_overlay_stamp`

**Usage**

```bash
export DEEP_PROJECT_NAME=gb200test
export DEEP_DATASET_PATH=/path/to/datasets

./BUILD-DOCKER-CONTAINER.sh train-gb200
./RUN-DOCKER-CONTAINER.sh train-gb200
# After exec, verify cu128 is in effect:
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 2.10.0+cu128 / 12.8
```

**Caveat**

The overlay-driven `LD_LIBRARY_PATH` is only applied in interactive login
shells. To run commands via `docker exec`, use `docker exec ... bash -lc
'...'` (login shell) or prefix your command with
`source ${UV_PROJECT_ENVIRONMENT}/.gb200_overlay_env.sh` (inside the container
this defaults to `/home/cache/venv`).

### 6.3 Docker: HSR deploy (`deploy`)

`Dockerfile.deploy` is Ubuntu 20.04 / ROS Noetic with the HSR catkin workspace
(hsr_data_msgs, hsr_data_collection, hsr_openpi_deploy).

**Build inputs**

```bash
export DEEP_PROJECT_NAME=hsrdeploy
export HSR_APT_USER=<apt user>      # HSR vendor apt credentials
export HSR_APT_PASSWORD=<apt pass>
./BUILD-DOCKER-CONTAINER.sh deploy
```

**Run**

```bash
# Resolve the robot: either ROBOT_NAME or HSR_IP
export ROBOT_NAME=hsrb107           # or:
# export HSR_IP=192.168.1.2

# ROS_IP is picked interactively (when multiple interfaces are detected)
./RUN-DOCKER-CONTAINER.sh deploy
```

Behavior of `RUN-DOCKER-COMMON.sh::openpi_resolve_deploy_network`:

- If `HSR_IP` is unset, resolve `ROBOT_NAME` via `getent hosts` then
  `avahi-resolve`
- If `ROS_IP` is unset, pick from `ifconfig` / `ip addr` output
- `ROS_MASTER_URI` is auto-set to `http://${HSR_IP}:11311`

On deploy mode, `initialize-ros-env.sh` also runs (catkin setup etc.).

---

## 7. Model deployment

On-robot inference is performed inside the `deploy` Docker container (§6.3)
using `roslaunch`. A separate `server/Dockerfile` is provided to run a
WebSocket-based policy server as a standalone process for external evaluation
tools or remote inference clients (§7.2).

### 7.1 On-robot deployment (ROS / `roslaunch`)

Prerequisites:

- Container built and running via §6.3; you are inside the container shell
- A trained checkpoint (step directory) is accessible from inside the
  container (e.g. `/home/openpi/checkpoints/...`)
- The HSR is network-reachable; `ROS_MASTER_URI` resolves to
  `http://${HSR_IP}:11311` (handled by §6.3)

#### 7.1.1 Load the ROS environment

Inside the container:

```bash
source /root/catkin_ws/devel/setup.bash
```

`initialize-docker-container.sh` writes this line into `.bashrc`, so login
shells (`bash -l` / `exec bash`) pick it up automatically.

#### 7.1.2 Sync the Python (uv) environment

```bash
cd /home/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
```

The container's first-boot hook (`warm_uv_environment`) already syncs once,
but doing it explicitly surfaces dependency drift early.

#### 7.1.3 Build the ROS packages

```bash
cd /root/catkin_ws
catkin build
source devel/setup.bash
```

Re-run after changes under `deploy/`.

#### 7.1.4 Start inference

Before launching, verify:

- `src/openpi/training/config.py` — tyro path only; check the `TrainConfig`
  you intend to use
- `deploy/hsr_data_collection/hsr_data_collection/config/hsr_data_collection_config.yaml`
  — controller selection (DualShock 3 vs 4)

Then connect a DualShock 3 / 4 controller to the HSR.

**A. Launch with a tyro config name**

```bash
roslaunch hsr_openpi hsr_openpi.launch \
  config_name:=pi05_hsr \
  checkpoint_dir:=/home/openpi/checkpoints/pi05_hsr/my_experiment/100000
```

**B. Launch with a YAML config (recommended)**

```bash
roslaunch hsr_openpi hsr_openpi.launch \
  config_yaml:=configs/experiments/your_run.yaml \
  checkpoint_dir:=/path/to/your_run/100000
```

Main launch arguments (`deploy/hsr_openpi_deploy/launch/hsr_openpi.launch`):

| Argument | Default | Purpose |
| --- | --- | --- |
| `config_name` | `pi0_hsr_low_mem_finetune` | Registered tyro `TrainConfig` name |
| `config_yaml` | `""` | Experiment YAML (takes precedence over `config_name`) |
| `checkpoint_dir` | (required) | Step directory to load |
| `update_freq` | `10` | Inference loop rate (Hz) |
| `adopted_action_chunks` | `10` | Number of action chunks consumed per step |
| `upsample`, `upsample_hz`, `upsample_method` | `true`, `100`, `spline` | Temporal action upsampling |
| `action_smoothing`, `ema_alpha`, `ma_window` | `ema`, `0.2`, `5` | Action smoothing (`ema` / `moving_average` / ...) |
| `smooth_gripper`, `smooth_base` | `false`, `false` | Whether to smooth gripper / base |
| `gripper_mode` | `hybrid` | Gripper control mode |
| `save_exec_trace` | `false` | Whether to persist the execution trace |

#### 7.1.5 Update the language instruction at runtime

```bash
rosservice call /hsr_openpi/update_instruction "message: 'Open the oven toaster'"
```

#### 7.1.6 Start action execution

Press the controller's **left D-pad** to start execution.

### 7.2 Policy server (`server/Dockerfile`)

`server/` at the repo root holds the assets for building a lean WebSocket
policy server as a standalone container. Use this when an external client
(evaluation tool, remote inference harness, etc.) needs to issue inference
requests over the network.

For the details (image build, `server/entrypoint.sh` environment variables,
checkpoint-embedded YAML auto-detection, hot reload / health check endpoints),
see `server/Dockerfile`, `server/entrypoint.sh`,
`server/serve_hsr_policy_ws.py`, and
`src/openpi/serving/websocket_policy_server.py`.

---

## 8. HPC — SLURM

### 8.1 Single node (`RUN-UV-SBATCH.sh` / `RUN-UV-SBATCH-8GPU.sh`)

Both are **tyro (`CONFIG_NAME`) path only**. They do not support YAML mode.

Minimum `.env` / `.env.local`:

```bash
# .env (experiment)
CONFIG_NAME=pi05_hsr        # a name registered in src/openpi/training/config.py
EXP_NAME=my_slurm_run

# .env.local (personal)
WANDB_API_KEY=...
DATA_DIR=/path/to/lerobot_datasets
HF_LEROBOT_HOME=/path/to/lerobot_cache
CHECKPOINT_DIR=/path/to/checkpoints
CACHE_ROOT=/path/to/cache_root
```

Submit:

```bash
sbatch RUN-UV-SBATCH.sh        # 1 node / 1 GPU / 24h
sbatch RUN-UV-SBATCH-8GPU.sh   # 1 node / 8 GPUs
```

Internally the script runs:

1. `module load cuda` (if Environment Modules is present)
2. `RUN-UV-COMMON.sh`: load `.env`, then `.env.local`
3. `pixi run uv sync`
4. `aggregate_stats_simple.py` to compute norm stats
5. `pixi run uv run scripts/train.py "${CONFIG_NAME}" --exp-name=... --overwrite`

### 8.2 Multi-node (`RUN-UV-SBATCH-OPENPI-MULTINODE.sh`)

YAML-mode multi-node launcher. The script re-submits itself via `sbatch` and
then uses `script_distribute_slurm_tasks.sh` to launch per-rank `srun` tasks.

Minimum `.env` / `.env.local`:

```bash
# .env (experiment)
OPENPI_CONFIG_YAML=configs/experiments/example_multinode.yaml
EXP_NAME=pi05_hsr_task689_run1

OPENPI_NUM_NODES=4
OPENPI_GRES=gpu:8
OPENPI_CPUS_PER_TASK=240
OPENPI_SBATCH_ARGS="--partition=<your-partition> --mem=1490000M --time=336:00:00"
OPENPI_RESUME=0
OPENPI_OVERWRITE=1

OPENPI_PREFLIGHT_ENABLE=1

# .env.local (personal)
WANDB_API_KEY=...
CACHE_ROOT=/groups/.../tmp_storage
CHECKPOINT_DIR=/groups/.../checkpoints
HF_LEROBOT_HOME=/groups/.../lerobot_cache
```

Submit:

```bash
bash RUN-UV-SBATCH-OPENPI-MULTINODE.sh
```

Keep YAML `gpu.num_gpus` aligned with `OPENPI_NUM_NODES * (GPUs per node from OPENPI_GRES)`;
the launcher warns when they disagree.

### 8.3 Slurm interactive allocation (`run_interactive_node.sh`)

A `salloc` wrapper for getting an interactive Slurm node.

```bash
./run_interactive_node.sh                      # defaults: 8 GPU / 192h / 1900G mem
./run_interactive_node.sh -g 1 -t 1:00:00      # 1 GPU / 1 hour
./run_interactive_node.sh -p <your-partition> -g 8 -t 4:00:00 -c 56
./run_interactive_node.sh --cpu-only           # no GPU (connector nodes)
```

Flags:

| Flag | Description |
| --- | --- |
| `-p, --partition PART` | Slurm partition (default `<your-partition>`) |
| `-g, --gpus N` | GPUs per node (default 8) |
| `-c, --cpus N` | CPUs per task (default 224) |
| `-t, --time HH:MM:SS` | Wall time (default `192:00:00`) |
| `-m, --mem SIZE` | Memory (default `1900G`) |
| `--cpu-only` | Drop the GPU request (connector nodes) |

Once you have a shell, the usual way to run training is to submit a
sbatch-based launcher (§8.1 / §8.2). The launchers invoke
`pixi run uv sync` / `pixi run uv run ...` internally.

---

## 9. HPC — PBS (qsub)

PBS + MPI (OpenMPI / HPCX) path, developed against ABCI.

### 9.1 Preferred: `RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh`

YAML-only. Loads HPCX via `module load` and calls `mpirun` directly within the
PBS job.

Minimum `.env` / `.env.local`:

```bash
# .env (experiment)
OPENPI_CONFIG_YAML=configs/experiments/example_multinode.yaml
EXP_NAME=pi05_hsr_task6_4nodes_run1
OPENPI_RESUME=1            # to continue from the latest checkpoint
OPENPI_OVERWRITE=0

# .env.local (personal)
WANDB_API_KEY=...
DATA_DIR=/groups/.../lerobot_datasets
HF_LEROBOT_HOME=/groups/...
CHECKPOINT_DIR=/groups/.../${USER}/AiroPi/checkpoints
CACHE_ROOT=/groups/.../tmp_storage
```

PBS headers at the top of the script fix `select` / `walltime`. Node count is
derived from `select=N:ncpus=192:mpiprocs=8` (assumes 8 GPUs per node;
`gpu.num_gpus` in the YAML should be `N * 8`).

Submit:

```bash
qsub RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh
# Or override the PBS headers:
qsub -l select=4:ncpus=192:mpiprocs=8 RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh
```

A healthy run logs from each rank:

```
[INFO] MASTER_ADDR=node-abc
[INFO] MASTER_PORT=12345
[INFO] NUM_NODES=4
[INFO] CONFIG=configs/experiments/example_multinode.yaml
```

Pre-steps (base-model download, norm-stat recompute via
`scripts/aggregate_stats_fast.py`) are executed by a single process before
`mpirun` (through `pixi run uv run python`).

### 9.2 Full-featured: `RUN-UV-QSUB-OPENPI-MULTINODE.sh`

Adds UCX tuning, preflight parquet checks, variable `OPENPI_MPI_TASKS_PER_NODE`,
etc. Relevant extra `.env` variables:

```bash
OPENPI_MPI_TASKS_PER_NODE=1
OPENPI_MPI_MODULE=hpcx/2.20
OPENPI_MPI_LAUNCHER=mpirun
OPENPI_MPI_MAP_BY=ppr:1:node
OPENPI_MPI_SHARED_TMPDIR=/groups/.../tmp_storage/mpi
OPENPI_MPI_ARGS="--bind-to none"
OPENPI_MPI_ASSIGN_UCX_NET_DEVICES=1
OPENPI_MPI_NDR_COUNT=8
UCX_MAX_RNDV_RAILS=8
UCX_MAX_EAGER_RAILS=2

OPENPI_QSUB_ARGS="-V -q <your-queue> -P <your-project> -l walltime=336:00:00"
OPENPI_QSUB_SELECT="select=4:mpiprocs=1"

OPENPI_PREFLIGHT_ENABLE=1
OPENPI_PREFLIGHT_SAMPLES_PER_NODE=20
OPENPI_PREFLIGHT_RETRIES=3
```

Submit:

```bash
bash RUN-UV-QSUB-OPENPI-MULTINODE.sh
```

### 9.3 Singularity

Fallback for HPC sites without Docker. `docker/openpi.def` defines the image
and `BUILD_SINGULARITY.sh` builds `airopi.sif`.

```bash
./BUILD_SINGULARITY.sh                        # produces airopi.sif
qsub RUN-SINGULARITY.sh                       # training + uv sync
qsub RUN-SINGULARITY-FT.sh                    # fine-tune
qsub RUN-SINGULARITY-COMPUTE-SIMPLE-NORM.sh   # norm stats only
```

- Each script supports `SIF_PATH` / `WORK_DIR` overrides via env / args
- `--bind /groups/<your-project>/dataset:/groups/<your-project>/dataset` is ABCI-specific;
  update the line for other sites (otherwise `singularity exec` fails)
- Env file loading follows Family B semantics (see §4.2) — if `.env.local`
  exists, `.env` is **not** read

---

## 10. Cross-cutting tips

### 10.1 Norm stats scripts

| Script | Characteristics | Use |
| --- | --- | --- |
| `scripts/aggregate_stats_fast.py` | Parallel JAX aggregation | HPC / multi-node |
| `scripts/aggregate_stats_simple.py` | Pure Python | Single-node, debugging |
| `scripts/compute_norm_stats.py` | tyro CONFIG_NAME entrypoint | tyro mode only |

### 10.2 Checkpoint directory layout

Orbax writes the following per save step:

```
${CHECKPOINT_DIR}/<exp_name>/
 └── <step>/
     ├── train_state/             # Optimizer state etc.
     ├── params/                  # Inference parameters
     ├── assets/<asset_id>/norm_stats.json
     └── experiment_config/experiment_config.yaml   # Only in YAML mode
```

`experiment_config.yaml` holds the merged-with-`_base_` YAML, so the policy
server (§7.2) can reconstruct the experiment semantics from a step directory
alone via its auto-detection path.

### 10.3 W&B integration

- Set `WANDB_API_KEY` in `.env.local` to enable
- Disable via YAML `experiment.wandb_enabled: false`
- In multi-node runs, only rank 0 logs; other ranks initialize wandb in
  `disabled` mode

### 10.4 Distributed environment resolution

`src/openpi/training/distributed.py` probes the following env vars in order.
MPI-provided variables are picked up automatically.

| Canonical | MPI fallbacks |
| --- | --- |
| `RANK` | `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `PMIX_RANK`, `MV2_COMM_WORLD_RANK` |
| `LOCAL_RANK` | `OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, ... |
| `WORLD_SIZE` | `OMPI_COMM_WORLD_SIZE`, ... |
| `LOCAL_WORLD_SIZE` | `OMPI_COMM_WORLD_LOCAL_SIZE`, ... |
| `MASTER_ADDR`, `MASTER_PORT` | Required explicitly |

### 10.5 When to use `pixi run uv run` vs bare `uv run`

Local training is pinned to the Docker flow (bare `uv run`), so you almost
never call `pixi run` directly.

- **Inside Docker / Singularity**: bare `uv run ...` (ffmpeg is in the image)
- **Inside HPC launchers**: `RUN-UV-COMMON.sh` invokes `pixi run uv sync` /
  `pixi run uv run ...` automatically. pixi and uv must be installed on the
  login node
- **Exception**: `RUN-UV-QSUB-OPENPI-MULTINODE.sh` (the full-featured variant)
  still uses bare `uv run` historically — candidate for cleanup

---

## 11. `.env` / `.env.local` cheatsheet

### 11.1 `.env` (experiment / launcher, mirrors `.env.example`)

| Variable | Scope | Description |
| --- | --- | --- |
| `OPENPI_CONFIG_YAML` | YAML path | Training YAML (relative or absolute) |
| `EXP_NAME` | All | Run identifier |
| `CONFIG_NAME` | tyro | Registered `TrainConfig` name (legacy) |
| `DATASET_NAME` | tyro | Dataset subpath (legacy) |
| `OPENPI_RESUME` | Training | `1` passes `--resume` (mutually exclusive with overwrite) |
| `OPENPI_OVERWRITE` | Training | `1` passes `--overwrite` |
| `OPENPI_NUM_NODES` | SLURM / PBS | Node count |
| `OPENPI_GRES` | SLURM | e.g. `gpu:8` |
| `OPENPI_SBATCH_ARGS` | SLURM | Extra SBATCH options |
| `OPENPI_CPUS_PER_TASK` | SLURM | CPUs per task (default 240) |
| `OPENPI_MPI_TASKS_PER_NODE` | PBS (full) | MPI ranks per node |
| `OPENPI_MPI_MAP_BY` | PBS (full) | OpenMPI map specification |
| `OPENPI_MPI_LAUNCHER` | PBS (full) | `mpirun` / `mpiexec` |
| `OPENPI_MPI_MODULE` | PBS (full) | Module to load (default `hpcx/2.20`). The SIMPLE variant hardcodes `hpcx/2.20` |
| `OPENPI_MPI_SHARED_TMPDIR` | PBS (full) | Shared MPI temp dir |
| `OPENPI_MPI_ARGS` | PBS (full) | Extra mpirun options |
| `OPENPI_QSUB_ARGS` | PBS (full) | Extra qsub options |
| `OPENPI_QSUB_SELECT` | PBS (full) | qsub `-l select=...` |
| `OPENPI_MPI_ASSIGN_UCX_NET_DEVICES` | PBS (full) | `1` assigns `UCX_NET_DEVICES` per rank |
| `OPENPI_MPI_NDR_COUNT` | PBS (full) | HCA count (default 8) |
| `OPENPI_AGGREGATE_STATS_ENABLE` | PBS (full) | `1` recomputes norm stats before training |
| `OPENPI_PREFLIGHT_ENABLE` | SLURM / PBS (full) | `1` enables parquet cross-node check |
| `OPENPI_PREFLIGHT_SAMPLES_PER_NODE` | SLURM / PBS (full) | Preflight sample count |

### 11.2 `.env.local` (personal / machine, mirrors `.env.local.example`)

| Variable | Scope | Description |
| --- | --- | --- |
| `WANDB_API_KEY` | All | W&B logging (logging is off when unset) |
| `WORKING_DIR` | uv | Working dir (default: script location) |
| `CACHE_ROOT` | Training | Base for uv / HF / tmp / XDG caches (writable from compute nodes) |
| `CHECKPOINT_DIR` | Training | Checkpoint root |
| `HF_LEROBOT_HOME` | Training | LeRobot HuggingFace cache |
| `DATA_DIR` | Training | LeRobot dataset root (usually inferred from YAML; override only when necessary) |

### 11.3 Docker variables

| Variable | Scope | Description |
| --- | --- | --- |
| `DEEP_PROJECT_NAME` | Docker | compose project name (required) |
| `DEEP_DATASET_PATH` | Docker | Host dataset path |
| `DEEP_CACHE_ROOT_HOST` | Docker | Host cache root (default `./.docker_cache`) |
| `DEEP_IMAGE_TAG` | Docker | Image tag (defaults to the mode name) |
| `DEEP_DOCKERFILE` | Docker | Dockerfile (auto-set by mode) |
| `OPENPI_PROJECT_PYTHON` | Docker | Default `3.11` |
| `OPENPI_ENABLE_ROS_INIT` | Docker | `auto` (default) / `1` / `0` |
| `OPENPI_PYTHON_OVERLAY_HOOK` | Docker (GB200) | Overlay script path (set by Dockerfile) |
| `HSR_APT_USER`, `HSR_APT_PASSWORD` | Docker deploy | HSR apt credentials (required) |
| `ROBOT_NAME`, `HSR_IP`, `ROS_IP`, `ROS_MASTER_URI` | Docker deploy | ROS connectivity |

### 11.4 Policy server (`server/`)

| Variable | Scope | Description |
| --- | --- | --- |
| `POLICY_CHECKPOINT_DIR` | server | Step directory for inference (required) |
| `POLICY_CONFIG_NAME`, `POLICY_CONFIG_YAML` | server | One of them (unless the checkpoint has an embedded `experiment_config.yaml`) |
| `POLICY_SERVER_HOST`, `POLICY_SERVER_PORT` | server | Default `0.0.0.0:8000` |
| `POLICY_DEFAULT_PROMPT` | server | Prompt fallback |
| `POLICY_RECORD_DIR` | server | Recording directory |
| `POLICY_PYTORCH_DEVICE` | server | `cuda` / `cpu` / ... |

### 11.5 Env-file overrides

| Variable | Scope | Description |
| --- | --- | --- |
| `HSR_OPENPI_ENV_FILE` | Family A and most of Family B | Override for the path of `.env` |
| `AIROPI_ENV_FILE` | `RUN-SINGULARITY-FT.sh` only | Same role, inconsistent naming |

---

## 12. Troubleshooting

### 12.1 `uv sync` fails inside the Docker container

- Network flakiness usually resolves on retry
- Pass `GIT_LFS_SKIP_SMUDGE=1` to `uv sync` if you don't want LeRobot LFS to
  pull
- A root-owned `.venv/` left over on the host can clash with the mounted
  repo. The Docker-side venv is isolated at `${UV_PROJECT_ENVIRONMENT}`
  (default `/home/cache/venv` = host `.docker_cache/venv`), so you can
  safely delete the host's `.venv/` if it was created by a previous
  container run

### 12.2 GB200 overlay not re-applied / cu128 not active

- `${UV_PROJECT_ENVIRONMENT}/.gb200_overlay_stamp` guards re-application.
  Changing overlay versions (`OPENPI_GB200_TORCH_VERSION`, ...) triggers a
  reinstall on the next container start. To force it, remove the stamp
  file and restart the container

### 12.3 Deploy container cannot resolve ROS

- When `ROBOT_NAME` is a `.local` name, `avahi-resolve -4 --name ...` must
  succeed (`apt install avahi-utils`)
- To force an IP, set `HSR_IP=...` in `.env` or the shell

---

## 13. Appendix

### 13.1 Tracked YAML templates

The only YAMLs committed under `configs/experiments/` are:

- `example_base.yaml` — minimal pi0.5 fine-tune
- `example_lora.yaml` — LoRA (vision / action) example
- `example_multinode.yaml` — multi-node GPU / scaling example

To share a new config in-repo, copy one of the templates and rename it with
the `example_` prefix (or update the `.gitignore` rule).

### 13.2 Python / key dependency versions

- Python 3.11.x
- jax 0.5.3 (cu12)
- torch 2.7.1 (non-Blackwell) / 2.10.0+cu128 (Blackwell overlay)
- flax 0.10.2, orbax-checkpoint 0.11.13
- transformers 4.53.2
- lerobot (pinned git rev, see `pyproject.toml`)
- pixi: ffmpeg ≥ 7 < 8, pkg-config, compilers, cython (conda-forge)

### 13.3 Released model weights

The following π0.5 checkpoints (HSR) are planned for public release. Each entry
corresponds to a step directory under `${CHECKPOINT_DIR}/<exp_name>/<step>/`
produced by `scripts/train.py` (layout described in §10.2), and can be loaded
either via `checkpoint_dir:=...` on the deploy launcher (§7.1) or by pointing
the policy server (§7.2) at the step directory.

| Stage | Run name (step) | Tasks |
| --- | --- | --- |
| Pre-training | `pi05_hsr_75tasks_fast_multinode_8nodes_vision_lora_action_full` (step 250,000) | 75 HSR tasks; π0.5 with vision LoRA + full action expert, trained on 8 nodes |
| Pre-training + post-training | `pi05_hsr_exercise_ph1_0405_lora_pretrain` (step 49,999) | Table pick-and-place (1 task) |
| Pre-training + post-training | `pi05_hsr_task6891011_level12_260304_finetune_68tasks_full_pi05` (step 200,000) | Bottle relocation, two-bottle relocation, box relocation, open/close microwave, cup relocation (5 tasks) |
