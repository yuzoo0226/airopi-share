# How to Deploy pi0

## 1. Installation
```bash
# Clone the main repository
git clone https://git.hsr.io/hsrtx/hsr_openpi.git

# Switch to the deployment branch
git switch feature/deploy
# clone submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
```

## 2. Environment setup
Set these variables before building or entering the deploy container.

```bash
export DEEP_PROJECT_NAME="your_project_name"
export ROBOT_NAME="hsrb107"
export ROS_IP="your_machine_ip"
export HSR_IP="robot_ip"
export HSR_APT_USER="your_hsr_package_user"
export HSR_APT_PASSWORD="your_hsr_package_password"
```

Notes:
- `HSR_APT_USER` and `HSR_APT_PASSWORD` are required build arguments for the deploy image. They are no longer embedded in tracked Dockerfiles.
- `RUN-DOCKER-CONTAINER.sh deploy` injects `ROS_IP`, `HSR_IP`, and `ROS_MASTER_URI` into each interactive `docker exec` session, so switching networks or robots does not leave stale values in `~/.bashrc`.

## 3. Build the Docker image
```bash
./BUILD-DOCKER-CONTAINER.sh deploy
```

## 4. Place the model weights into a directory
For example, place them under `checkpoints/`.

## 5. Launch the Docker container
```bash
./RUN-DOCKER-CONTAINER.sh deploy
```

## 6. Sync the Python environment
The OpenPI project environment is pinned to Python 3.11 inside `/home/openpi/.venv`. The ROS base layer remains on Ubuntu 20.04's system Python for ROS Noetic compatibility.

```bash
cd /home/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
```

The container initialization script also warms this environment automatically on first entry when the cache directories and network are available.

## 7. Quick inference smoke test
This verifies the policy server and client pipeline only. It does not open a simulator window.

Terminal 1:
```bash
cd /home/openpi
uv run scripts/serve_policy.py --env ALOHA_SIM
```

Terminal 2:
```bash
cd /home/openpi
uv run examples/simple_client/main.py --env ALOHA_SIM
```

Notes:
- The first run downloads a checkpoint from `gs://openpi-assets`, so network access is required.
- This checks only the policy server and client path.

## 8. Quick training smoke test
```bash
cd /home/openpi
uv run scripts/train.py debug --exp_name debug_smoke --overwrite
```

## 9. ROS logging patch
The standard Docker initialization flow applies the known `rosgraph/roslogging.py` stack workaround automatically. Manual edits are no longer required.

## 10. Build ROS packages
```bash
cd /root/catkin_ws
catkin build
source devel/setup.bash
```

## 11. Run model inference
Before launching, confirm the following files are configured for your checkpoint and controller:
- `src/openpi/training/config.py`
- `deploy/hsr_data_collection/hsr_data_collection/config/hsr_data_collection_config.yaml`

Example launch with an explicit config name:
```bash
roslaunch hsr_openpi hsr_openpi.launch \
  config_name:=pi0_hsr_weblab_leader \
  checkpoint_dir:=/home/openpi/checkpoints/pi0_hsr_weblab_leader/my_experiment/99
```

You can also load the deployment model definition directly from an experiment YAML:
```bash
roslaunch hsr_openpi hsr_openpi.launch \
  config_yaml:=configs/experiments/pi05_hsr_task6891011_level12_260304_finetune.yaml \
  checkpoint_dir:=/home/datasets/pi05_hsr_task6891011_level12_260304_finetune/pi05_hsr_task6891011_level12_260304_fullfinetune/100000
```

To change the language instruction at runtime:
```bash
rosservice call /hsr_openpi/update_instruction "message: 'Open the oven toaster'"
```

## 12. Start model action execution
Press the left directional button on the controller to start the action.

## 13. Evaluate from `airoa-evaluation-pipeline`
This repository includes a top-level `server/` directory so `airoa-evaluation-pipeline` can use this repo directly as `worktree_path`.

The evaluator builds the model server from:

```text
<worktree_path>/server/Dockerfile
```

An example evaluator model definition is stored at:

```text
deploy/models.airoa.example.json
```

Example entry:

```json
{
  "name": "pi05_hsr_68tasks_full",
  "worktree_path": "/path/to/AiroPi_task_sampler",
  "checkpoint_path": "/abs/path/to/checkpoints/pi05_hsr_task6891011_level12_260304_finetune_68tasks_full_pi05/100000",
  "config_name": "pi05_hsr_task6891011_level12_260304_finetune_68tasks_full_pi05",
  "env": {
    "POLICY_CONFIG_YAML": "/workspace/configs/experiments/pi05_hsr_task6891011_level12_260304_finetune_68tasks_full_pi05.yaml",
    "POLICY_PYTORCH_DEVICE": "cuda"
  }
}
```

Notes:
- `checkpoint_path` must point to the checkpoint step directory used for inference.
- `config_name` remains required by the evaluator JSON schema, but when `POLICY_CONFIG_YAML` is set, the server loads the architecture from YAML and uses `config_name` only as metadata.
- `POLICY_CONFIG_YAML` must be a path inside the built container. Because the repo is copied to `/workspace`, use paths like `/workspace/configs/experiments/...yaml`.

### YAML mapping for current HSR finetune variants
- `full`: `/workspace/configs/experiments/pi05_hsr_task6891011_level12_260304_finetune_68tasks_full_pi05.yaml`
- `action_only`: `/workspace/configs/experiments/pi05_hsr_task6891011_level12_260304_finetune_68tasks_action_only_pi05.yaml`
- `lora`: `/workspace/configs/experiments/pi05_hsr_task6891011_level12_260304_finetune_68tasks_lora_pi05.yaml`
- `head_only`: `/workspace/configs/experiments/pi05_hsr_task6891011_level12_260304_finetune_68tasks_head_only_pi05.yaml`

### Evaluator command
Run this in `~/icra_evaluation/airoa-evaluation-pipeline`:

```bash
python3 deploy/eval_competition_web.py \
  --models deploy/models.json \
  --tasks deploy/evaluation_tasks.json \
  --runs-per-sht 5 \
  --hsr-ip <HSR_IP> \
  --host 0.0.0.0 \
  --port 8080 \
  --ros-ip-choice 2
```
