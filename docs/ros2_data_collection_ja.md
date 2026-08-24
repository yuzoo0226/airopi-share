# ROS 2 / Gazebo でのデータ収集 → 変換 → 学習

Ignition Gazebo 上の HSR を動かして ROS 2 bag を記録し、LeRobot データセットに
変換して pi0.5 をファインチューンするまでの手順です。

推論環境の構築は [`ros2_deploy_ja.md`](ros2_deploy_ja.md)、
実際の構築記録とハマった点は [`ros2_worklog_ja.md`](ros2_worklog_ja.md) を参照してください。

---

## 1. 全体像

```
 ① 収集                     ② 変換                      ③ 統計 → ④ 学習 → ⑤ 推論
 ros2 bag (mcap)  ──▶  LeRobot v2.1 dataset  ──▶ norm_stats ──▶ checkpoint ──▶ Gazebo
 hsr_openpi/                rosbag2_to_lerobot.py     aggregate_    train.py     hsr_openpi
 collect_data.launch.py                               stats_fast.py              .launch.py
```

| 段階 | コンテナ | 実行するもの |
| --- | --- | --- |
| ① 収集 | `hsr-ros2-sim` | `ros2 launch hsr_openpi collect_data.launch.py` |
| ② 変換 | `airopi_ros2_deep_1`（学習用） | `deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py` |
| ③ 統計 | 学習用 | `scripts/aggregate_stats_fast.py` |
| ④ 学習 | 学習用 | `scripts/train.py --config-yaml ...` |
| ⑤ 推論 | `hsr-ros2-sim` + `airopi-policy-server` | `ros2 launch hsr_openpi hsr_openpi.launch.py` |

---

## 2. ① Gazebo でのデータ収集

```bash
# シミュレータを起動しておく
docker compose exec hsr-sim bash -lc \
  'ros2 launch hsr_openpi hsr_sim.launch.py world:=apartment robot_pos_x:=5.0 robot_pos_y:=6.6'

# ランダム動作 + bag 記録
docker compose exec hsr-sim bash -lc \
  'ros2 launch hsr_openpi collect_data.launch.py \
     bag_path:=/home/hsr/hsr_ros2_ws/_bags/random_01 \
     num_episodes:=6 episode_duration:=30.0 reset_duration:=5.0 \
     seed:=1 task:="move the arm around randomly"'
```

`collect_data.launch.py` は 3 つを同時に起動します。

1. `image_transport republish`（raw → compressed）。
   Ignition は raw 画像しか出さないため、これを挟むことで実機と同じ
   `.../compressed` トピック名で記録でき、bag のサイズも約 1/30 になります。
2. `hsr_random_motion` ノード。
   各関節を **Ornstein-Uhlenbeck 過程**（平均回帰つきランダムウォーク）で
   動かします。URDF の可動域から安全マージンを取った範囲にクランプし、
   速度も制限しているため、ホワイトノイズ的な指令にならず学習データとして
   使える連続的な軌道になります。台車も小さな速度で動かします。
3. `ros2 bag record`（mcap）。

エピソード管理は自動です。`episode_duration` 秒動かしたあと `reset_duration`
秒かけて初期姿勢に戻り、次のエピソードに移ります。境界は
`/hsr_random_motion/episode`（`start <idx> <task>` / `end <idx>`）に publish され、
`/control_mode` も実機と同じく `auto` / `reset` / `stopped` が流れます。

> **重要**: `num_episodes` に到達すると `hsr_random_motion` は**プロセスごと終了**し、
> launch が `ros2 bag record` に SIGINT を送ります。rosbag2 は SIGINT を受けたときに
> だけ `metadata.yaml` を書くため、レコーダを `kill -9` すると
> `ros2 bag info` も変換スクリプトも開けない bag ができあがります。
> そうなってしまった場合は `ros2 bag reindex <dir> -s mcap` で復旧できます。

記録トピック（`collect_data.launch.py` の `RECORD_TOPICS`）:

```
/clock                                          /joint_states
/head_rgbd_sensor/rgb/image_rect_color/compressed
/head_rgbd_sensor/rgb/camera_info
/hand_camera/image_raw/compressed               /hand_camera/camera_info
/arm_trajectory_controller/joint_trajectory     /head_trajectory_controller/joint_trajectory
/gripper_controller/joint_trajectory            /omni_base_controller/cmd_vel
/odom                                           /tf  /tf_static
/control_mode                                   /hsr_random_motion/episode
```

---

## 3. ② bag → LeRobot データセット

```bash
# 学習コンテナ側から見える場所に bag を置く（compose の /home/bags もしくは /home/datasets）
docker exec airopi_ros2_deep_1 bash -lc '
  cd /home/openpi &&
  /home/cache/venv/bin/python deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py \
      --bag /home/bags/random_01 \
      --repo-id lerobot_datasets/hsr_gazebo_random \
      --root /home/datasets \
      --fps 10 --overwrite \
      --task "move the arm around randomly"'
```

事前に `uv pip install --python /home/cache/venv/bin/python rosbags` が必要です
（`rosbags` は純 Python の rosbag2 リーダで、ROS のインストールを必要としません）。

### 3.1 出力されるカラム

`LeRobotHSRDataConfig`（`action_mode: relative`）が読むものと一致します。

| カラム | dtype / shape | 意味 |
| --- | --- | --- |
| `observation.image.head` | image (480,640,3) | 頭部 RGBD センサのカラー画像 |
| `observation.image.hand` | image (480,640,3) | ハンドカメラ |
| `observation.state` | float32 (8,) | 実測関節角 |
| `action.relative` | float32 (11,) | 指令値（下記） |

`observation.state` の並びは
`[arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, hand_motor, head_pan, head_tilt]`。

`action.relative` の定義は**推論ノードの逆演算**です。

```python
action[0:5]  = arm_command  - state[0:5]     # アーム: 実測からの相対
action[5]    = gripper_command               # グリッパ: 絶対
action[6:8]  = head_command  - state[6:8]    # ヘッド: 実測からの相対
action[8:11] = cmd_vel (vx, vy, wz)          # 台車: 速度
```

推論ノード側は `command = action + [state[:5], 0, state[6:8], 0, 0, 0]` を行うので、
この定義で学習したポリシーはそのまま `hsr_openpi_node` で実行できます。

### 3.2 時刻の扱い

rosbag2 はメッセージに**記録 PC の実時刻**を打ちますが、シミュレータは `/clock`
で動いています。ヘッダを持つメッセージ（JointState / CompressedImage /
JointTrajectory）はヘッダの stamp がシミュレーション時刻なのでそれを使い、
ヘッダを持たないメッセージ（Twist / String）は記録済みの `/clock` から
実時刻 → シミュレーション時刻の写像を作って補正しています。

### 3.3 画像のチャネル順

`--image-order`（既定 `bgr`）で選びます。**推論側の `policy_image_order` と
必ず一致させてください**（既定同士は一致しています）。背景は
[`ros2_deploy_ja.md`](ros2_deploy_ja.md) 6.1 を参照。

### 3.4 画像の格納形式

`use_videos=False` で作るため、フレームは PNG として parquet 内に
`struct<bytes, path>` で格納されます。openpi の parquet バックエンドは
`_decode_parquet_image()` でこの形式をデコードできるので、動画コーデック
（torchcodec / ffmpeg）は学習時にも一切不要です。

---

## 4. ③ 正規化統計

```bash
docker exec airopi_ros2_deep_1 bash -lc '
  cd /home/openpi &&
  JAX_PLATFORMS=cpu /home/cache/venv/bin/python scripts/aggregate_stats_fast.py \
      --episodes-stats /home/datasets/lerobot_datasets/hsr_gazebo_random/meta/episodes_stats.jsonl \
      --output-file assets/example_hsr_gazebo_ros2/lerobot_datasets/hsr_gazebo_random/norm_stats.json \
      --chunk-dir /home/datasets/lerobot_datasets/hsr_gazebo_random/data/ \
      --action-column action.relative \
      --action-mode relative'
```

出力先は YAML の `assets_dir` / `asset_id` と揃える必要があります。

---

## 5. ④ 学習

設定例: [`configs/experiments/example_hsr_gazebo_ros2.yaml`](../configs/experiments/example_hsr_gazebo_ros2.yaml)
（1 GPU / batch 4 / 200 step の動作確認用。実運用ではステップ数とバッチを上げてください）

公開ベースモデルから始めるため `weight_loader.params_path` に
`/home/checkpoints/airoa-pi05-hsr-base/params` を指定しています。
`/home/checkpoints` は `docker/ros2/docker-compose.train.yml`（override）で
マウントされます。

```bash
export DEEP_PROJECT_NAME=airopi_ros2 CONTAINER=airopi_ros2_deep_1
export DEEP_IMAGE_TAG=train-gb200 DEEP_DOCKERFILE=./docker/Dockerfile.train.gb200
export DEEP_DATASET_PATH=<...>/datasets DEEP_CACHE_ROOT_HOST=<...>/.docker_cache
export DEEP_CHECKPOINT_PATH=<...>/checkpoints DEEP_BAG_PATH=<...>/hsr_ros2_ws/_bags
docker compose -p "$DEEP_PROJECT_NAME" \
    -f docker/docker-compose.yml -f docker/ros2/docker-compose.train.yml up -d

# sm_121 (GB10) の場合のみ: jax を 0.6.2 に上げる
docker exec "$CONTAINER" bash /home/openpi/scripts/ros2/apply_sm121_jax_overlay.sh

docker exec "$CONTAINER" bash -lc '
  cd /home/openpi &&
  CHECKPOINT_DIR=/home/checkpoints/_train \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  /home/cache/venv/bin/python scripts/train.py \
      --config-yaml configs/experiments/example_hsr_gazebo_ros2.yaml \
      --exp-name hsr_gazebo_ros2_smoke --overwrite'
```

> **`uv run` は使わないでください。** `uv run`（`--no-sync` なし）は環境を
> `uv.lock` に同期し直すため、GB200 の cu128 torch オーバーレイも
> 上記 jax オーバーレイも巻き戻されます。`${UV_PROJECT_ENVIRONMENT}/bin/python`
> を直接呼ぶか、`uv run --no-sync` を使ってください。

出力（`${CHECKPOINT_DIR}/<config名>/<exp-name>/<step>/`）:

```
199/
├── params/                                     # 推論用パラメータ
├── train_state/
├── assets/<asset_id>/norm_stats.json
└── experiment_config/experiment_config.yaml    # サーバが自動検出する
```

---

## 6. ⑤ 学習したモデルで推論

```bash
cd docker/ros2
POLICY_CHECKPOINT_DIR=/home/openpi/checkpoints/_train/example_hsr_gazebo_ros2/hsr_gazebo_ros2_smoke/199 \
POLICY_DEFAULT_PROMPT="move the arm around randomly" \
  docker compose up -d --force-recreate openpi-server

docker compose exec hsr-sim bash -lc \
  'ros2 launch hsr_openpi hsr_openpi.launch.py auto_start:=true \
     instruction:="move the arm around randomly"'
```

チェックポイントに `experiment_config.yaml` と `assets/.../norm_stats.json` が
同梱されているため、サーバ側で config 名を指定する必要はありません。

---

## 7. 実測値（GB10 / DGX Spark, apartment world）

| 項目 | 値 |
| --- | --- |
| 収集 | 6 エピソード × 30 s、bag 381 MiB（mcap, 229 s, 153k msgs） |
| 変換 | 6 エピソード / 1803 フレーム（10 fps）→ 405 MB |
| 学習 | 200 step / batch 4、`step_time≈0.32 s`、`data_time≈0.007 s` |
| loss | 0.364 → 0.24、`val_loss` 0.241 → 0.220 |
| チェックポイント | 8.1 GB（params + train_state） |

---

## 8. 既知の制約

* **ハンドカメラのレンダリングレートが低い**（empty world で 6〜9 Hz、
  apartment world でも 10 Hz 前後）。10 fps のデータセットだと同じ画像が
  複数フレームに使われることがあります。`--max-age` で古すぎるフレームは
  落としています。頭部・ステレオ・head_center カメラを URDF から外して
  レンダリング負荷を下げると改善します。
* **ランダム動作は「タスク」ではありません**。パイプライン検証用です。
  実際の学習にはテレオペや台本付きの動作でデータを取ってください。
  `hsr_random_motion` を差し替えるだけで同じ収集・変換経路が使えます。
* **`action.relative` の台車成分は指令速度**です。オドメトリからの実測値では
  ないため、台車が指令に追従しない状況（衝突など）では乖離します。
