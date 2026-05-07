### Docker Setup

This repository provides three maintained Docker entry points:
- `./BUILD-DOCKER-CONTAINER.sh train` and `./RUN-DOCKER-CONTAINER.sh train` for local training on the standard image
- `./BUILD-DOCKER-CONTAINER.sh train-gb200` and `./RUN-DOCKER-CONTAINER.sh train-gb200` for the GB200 overlay image
- `./BUILD-DOCKER-CONTAINER.sh deploy` and `./RUN-DOCKER-CONTAINER.sh deploy` for the ROS deployment image

### Prerequisites
- Install Docker Engine rather than the `snap` package or Docker Desktop when using NVIDIA GPUs.
- Install the NVIDIA container toolkit so the container can access CUDA.
- Rootless Docker is supported, but the host paths below must still be readable and writable by your user.

### Common environment variables
Set these before building or running containers.

```bash
export DEEP_PROJECT_NAME="airopi"
export DEEP_DATASET_PATH="/abs/path/to/datasets"
export DEEP_CACHE_ROOT_HOST="/abs/path/to/cache"
```

Optional overrides:

```bash
export DEEP_IMAGE_TAG="custom-tag"
export DEEP_DOCKERFILE="Dockerfile.train"
export OPENPI_PROJECT_PYTHON="3.11"
```

Notes:
- `DEEP_DATASET_PATH` is mounted at `/home/datasets`.
- `DEEP_CACHE_ROOT_HOST` is mounted into the container cache root and is used for Hugging Face, uv, pip, and wandb caches.
- `OPENPI_PROJECT_PYTHON` controls the Python version requested by `uv sync`. The maintained default is `3.11`.

### Local training workflow
Build and enter the standard training image:

```bash
./BUILD-DOCKER-CONTAINER.sh train
./RUN-DOCKER-CONTAINER.sh train
```

Inside the container:

```bash
cd /home/openpi
uv run scripts/train.py debug --exp_name debug_smoke --overwrite
```

On first entry, `docker/scripts/initialize-docker-container.sh` performs idempotent setup:
- creates cache directories
- warms the `uv` environment at Python 3.11
- adds `/home/openpi/.venv/bin` to the shell `PATH` when present
- applies the GB200 overlay hook only in `train-gb200` mode

Notes:
- `python3 -V` may still show the base OS Python (for example Ubuntu 22.04 ships Python 3.10), while `.venv/bin/python -V` and `uv run python -V` should resolve to the project Python requested by `OPENPI_PROJECT_PYTHON`.
- The training images install FFmpeg 7 so that PyAV (`av`) can be built from source during `uv sync` / `uv run`.

### GB200 workflow
Use the GB200 image when you need the PyTorch and video backend overlay configured by `docker/Dockerfile.train.gb200`:

```bash
./BUILD-DOCKER-CONTAINER.sh train-gb200
./RUN-DOCKER-CONTAINER.sh train-gb200
```

### Deploy workflow
The deploy image needs additional ROS and HSR package credentials:

```bash
export ROBOT_NAME="hsrb107"
export HSR_IP="robot_ip"
export ROS_IP="your_machine_ip"
export HSR_APT_USER="your_hsr_package_user"
export HSR_APT_PASSWORD="your_hsr_package_password"

./BUILD-DOCKER-CONTAINER.sh deploy
./RUN-DOCKER-CONTAINER.sh deploy
```

Notes:
- `HSR_APT_USER` and `HSR_APT_PASSWORD` are passed as Docker build arguments and are not stored in tracked Dockerfiles anymore.
- `RUN-DOCKER-CONTAINER.sh deploy` injects `ROS_IP`, `HSR_IP`, and `ROS_MASTER_URI` into the interactive `docker exec` session instead of mutating `~/.bashrc`.
- The ROS base image stays on Ubuntu 20.04's system Python for ROS Noetic compatibility. OpenPI project code is expected to run from the Python 3.11 environment in `/home/openpi/.venv`.
- The deploy image also installs FFmpeg 7 so `uv run` can build PyAV from source when LeRobot dependencies are present.

### Compose configuration check
To inspect the resolved Compose configuration without building, run:

```bash
HSR_APT_USER=dummy HSR_APT_PASSWORD=dummy docker compose -f docker/docker-compose.yml config
```
