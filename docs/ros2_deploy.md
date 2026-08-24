# HSR deployment on ROS 2 (Humble) + Gazebo simulation

How to run pi0 / pi0.5 policies on the HSR under **ROS 2 Humble**, either on the
real robot or in the open source Ignition Gazebo simulator. This is a port of the
ROS 1 Noetic deployment (`deploy/hsr_openpi_deploy`) with an unchanged control
loop.

> 日本語版: [`ros2_deploy_ja.md`](ros2_deploy_ja.md)
> Build log / known pitfalls (JA): [`ros2_worklog_ja.md`](ros2_worklog_ja.md)
> Data collection & training (JA): [`ros2_data_collection_ja.md`](ros2_data_collection_ja.md)

---

## 1. Architecture

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│ hsr-sim container            │        │ openpi-server container       │
│  ROS 2 Humble / Python 3.10  │  ws    │  Python 3.11 + JAX            │
│                              │◄──────►│                               │
│  Ignition Gazebo Fortress    │ :8010  │  serve_hsr_policy_ws.py       │
│  + HSR (hsr-project/*)       │        │  (pi0.5 checkpoint)           │
│  + hsr_openpi node           │        │                               │
└──────────────────────────────┘        └───────────────────────────────┘
```

ROS 2 Humble ships Python 3.10 while openpi needs Python 3.11 + JAX, so the
policy runs in its own process and is reached over a websocket. Passing
`policy_backend:=local` loads openpi in-process instead.

| Component | Where |
| --- | --- |
| Simulator | [hsr-project](https://github.com/hsr-project) `humble` branches (Ignition Gazebo Fortress) |
| ROS 2 node | `deploy/hsr_openpi_ros2/hsr_openpi` |
| Service definition | `deploy/hsr_openpi_ros2/hsr_openpi_msgs` |
| Policy server | existing `server/serve_hsr_policy_ws.py` + `docker/ros2/Dockerfile.server` |
| Base model | [`airoa-org/airoa-pi05-hsr-base`](https://huggingface.co/airoa-org/airoa-pi05-hsr-base) |

---

## 2. Setup

```bash
# 1. HSR simulator workspace, created next to this repository
./scripts/ros2/setup_hsr_ros2_ws.sh

# 2. base model (~6.6 GB) -> ../checkpoints/airoa-pi05-hsr-base
./scripts/ros2/download_base_model.sh

# 3. images + workspace build
cd docker/ros2
./build-sim-image.sh
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build openpi-server
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d hsr-sim
./build-workspace.sh
```

`setup_hsr_ros2_ws.sh` clones the 22 `hsr-project` repositories plus `ros_gz` and
`gz_ros2_control`, which have no arm64 debian packages and are therefore built
from source. `hsrb_robot_launch` (real robot only) and `tmc_pgr_camera`
(proprietary SDK) get a `COLCON_IGNORE`.

The released checkpoint embeds `experiment_config/experiment_config.yaml`, so the
policy server does not need `--config-name` / `--config-yaml`.

---

## 3. Running

```bash
# simulator (headless by default)
docker compose exec hsr-sim bash
ros2 launch hsr_openpi hsr_sim.launch.py
# ros2 launch hsr_openpi hsr_sim.launch.py world:=apartment_no_objects \
#     robot_pos_x:=5.0 robot_pos_y:=6.6
# ros2 launch hsr_openpi hsr_sim.launch.py headless:=false     # needs DISPLAY

# policy server
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d openpi-server
docker compose logs -f openpi-server

# inference node
docker compose exec hsr-sim bash
ros2 launch hsr_openpi hsr_openpi.launch.py \
    instruction:="Grasp the bottle." auto_start:=true
```

Runtime control:

```bash
ros2 service call /hsr_openpi/update_instruction hsr_openpi_msgs/srv/StringTrigger \
    "{message: 'Open the oven toaster'}"
ros2 service call /hsr_openpi/start std_srvs/srv/Trigger
ros2 service call /hsr_openpi/stop  std_srvs/srv/Trigger
ros2 launch hsr_openpi reset_pose.launch.py
```

`headless:=true` starts `gz sim -s --headless-rendering`, which renders the
cameras through EGL and needs no X server.

---

## 4. Topic map

The simulator and the real robot disagree on every name, so all of them are ROS
parameters selected by `robot_profile:=sim|real` (`config/{sim,real}_topics.yaml`).

| Purpose | Real robot (ROS 1 / hsrb bringup) | Ignition Gazebo (hsrb_gazebo_bringup) |
| --- | --- | --- |
| head camera | `/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed` | `/head_rgbd_sensor/rgb/image_rect_color` (raw) |
| hand camera | `/hsrb/hand_camera/image_raw/compressed` | `/hand_camera/image_raw` (raw) |
| joint states | `/hsrb/joint_states` | `/joint_states` |
| arm command | `/hsrb/arm_trajectory_controller/command` | `/arm_trajectory_controller/joint_trajectory` |
| head command | `/hsrb/head_trajectory_controller/command` | `/head_trajectory_controller/joint_trajectory` |
| gripper command | `/hsrb/gripper_controller/command` | `/gripper_controller/joint_trajectory` |
| base command | `/hsrb/command_velocity` (Twist) | `/omni_base_controller/cmd_vel` (Twist) |
| grasp action | `/hsrb/gripper_controller/grasp` | `/gripper_controller/grasp` |

Note that a ROS 2 `joint_trajectory_controller` listens on `~/joint_trajectory`,
not on `~/command`.

---

## 5. Launch arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `robot_profile` | `sim` | Topic layout (`sim` / `real`) |
| `policy_backend` | `websocket` | `websocket` (separate process) / `local` (in-process) |
| `policy_host` / `policy_port` | `127.0.0.1` / `8010` | Policy server endpoint |
| `checkpoint_dir` / `config_yaml` | `""` | Only used with `policy_backend:=local` |
| `instruction` | `Grasp the bottle.` | Initial language instruction |
| `update_freq` | `10` | Action chunk rate [Hz] |
| `adopted_action_chunks` | `10` | Chunk elements consumed per inference |
| `upsample` / `upsample_hz` / `upsample_method` | `true` / `100` / `spline` | Interpolation to the execution rate |
| `action_smoothing` / `ema_alpha` / `ma_window` | `ema` / `0.2` / `5` | Smoothing |
| `gripper_mode` | `hybrid` | `continuous` / `discrete` / `hybrid` |
| `policy_image_order` | `bgr` | Channel order handed to the policy (see §6.1) |
| `require_control_mode` / `auto_start` | `true` / `false` | Wait for `/control_mode == auto` |
| `save_exec_trace` | `false` | Save the execution trace (npz + png) |

---

## 6. Implementation notes

### 6.1 Image channel order (careful)

Training and deployment disagree on the channel order.

**Training produces RGB.** Following the three conversion steps in
`deploy/hsr_data_collection/conversion/`:

```python
# rosbag2pkl.py : received JPEG -> RGB array
img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, ::-1]
# rosbag2pkl.py : flipped back to BGR before writing => visually correct JPEG
cv2.imwrite(path, image[:, :, ::-1])
# pkl2np.py / pkl2rlds.py : read back and flip to RGB
image = cv2.imread(impath)[:, :, ::-1]
```

**Deployment feeds BGR.** The ROS 1 node
(`deploy/hsr_openpi_deploy/scripts/hsr_openpi.py`) passes the raw
`cv2.imdecode` output:

```python
image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, :]    # stays BGR
```

and the ROS 2 client of the ICRA evaluation runtime
(`tamukohlaboratory/airoa-evaluation-ICRA`) likewise converts `rgb8` frames to
BGR before inference.

The source order also depends on the transport: `cv2.imdecode` of a
`CompressedImage` yields BGR, while Ignition Gazebo publishes raw `rgb8`. The
node therefore normalises every frame to `policy_image_order` (default `bgr`)
instead of just flipping conditionally — otherwise the simulator would feed RGB
while the real robot feeds BGR.

`bgr` is the default because that is what the two deployments validated against
the released checkpoints do. Set `policy_image_order:=rgb` to match the training
pipeline instead; which one performs better is worth an A/B run.

### 6.2 JAX version

`pyproject.toml` pins `jax[cuda12]==0.5.3`, but the XLA inside jaxlib 0.5.3 does
not know sm_121 (NVIDIA GB10 / DGX Spark) and aborts while compiling any
float16 / bfloat16 kernel:

```
Unsupported conversion from bf16 to f16
LLVM ERROR: Unsupported rounding mode for conversion.
```

pi0.5 hits this immediately because its params are restored as bfloat16.
`docker/ros2/Dockerfile.server` therefore installs a jaxlib that supports
sm_121; override with `--build-arg JAX_VERSION=...` (0.5.3 is fine on x86
Ampere/Hopper).

### 6.3 Why the server image is small

`docker/ros2/Dockerfile.server` deliberately omits lerobot, torch, torchcodec and
FFmpeg 7: the released HSR checkpoints are JAX/orbax and
`server/serve_hsr_policy_ws.py` never imports the LeRobot data pipeline. This
avoids building PyAV against FFmpeg 7 from source on aarch64. Use
`docker/Dockerfile.train*` for training as before.

### 6.4 Simulator on arm64

* The ignition packages mirrored on `packages.ros.org` are not self consistent on
  arm64 (`ignition-fortress` requires `libignition-sensors6 >= 6.8.1`, only 6.8.0
  is mirrored), so the image adds `packages.osrfoundation.org` with an apt pin.
* `ros-humble-ros-gz-sim` and `ros-humble-gz-ros2-control` have no arm64 debian
  packages and are built from source in `src/_extern`.

---

## 7. Smoke testing without a policy server

```bash
# terminal 1, inside the hsr-sim container
python3 /home/hsr/airopi-share/deploy/hsr_openpi_ros2/tools/mock_policy_server.py \
    --port 8010 --pattern wiggle

# terminal 2
ros2 launch hsr_openpi hsr_openpi.launch.py auto_start:=true
```

If the arm, head and base start oscillating, the whole path — observation intake,
inference, upsampling, smoothing, controller commands — is wired up correctly.

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Waiting for observations: head_rgb(...)` never clears | Check the sensors with `ros2 topic hz`; headless rendering needs EGL (`NVIDIA_DRIVER_CAPABILITIES=all`) |
| `Action NOT executed (control mode is not active)` | Use `auto_start:=true` or call `/hsr_openpi/start` |
| Node blocks waiting for `/clock` | Gazebo is not running, or set `use_sim_time:=false` |
| Cannot reach the policy server | Port clash (8010 by default); keep `POLICY_SERVER_PORT` and `policy_port` in sync |
| `Unsupported rounding mode for conversion` | See §6.2, upgrade JAX |
| Out of memory | On unified-memory machines (GB10) free the GPU memory held by other processes (e.g. vLLM) |
