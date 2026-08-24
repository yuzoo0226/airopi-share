# ROS 2 (Humble) での HSR デプロイ / Gazebo シミュレーション

pi0 / pi0.5 ポリシーを **ROS 2 Humble** の HSR（実機・Ignition Gazebo シミュレータ）で
動かすための手順です。ROS 1 Noetic 版（`deploy/hsr_openpi_deploy`）の移植で、
制御ループの挙動は同じです。

> 英語版: [`ros2_deploy.md`](ros2_deploy.md)
> 構築記録・ハマった点: [`ros2_worklog_ja.md`](ros2_worklog_ja.md)
> データ収集・学習: [`ros2_data_collection_ja.md`](ros2_data_collection_ja.md)

---

## 1. 構成

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│ hsr-sim コンテナ              │        │ openpi-server コンテナ         │
│  ROS 2 Humble / Python 3.10  │  ws    │  Python 3.11 + JAX            │
│                              │◄──────►│                               │
│  Ignition Gazebo Fortress    │ :8010  │  serve_hsr_policy_ws.py       │
│  + HSR (hsr-project/*)       │        │  (pi0.5 チェックポイント)       │
│  + hsr_openpi ノード          │        │                               │
└──────────────────────────────┘        └───────────────────────────────┘
```

ROS 2 Humble は Python 3.10、openpi は Python 3.11 + JAX が必要なため、
**ポリシーを別プロセス（別コンテナ）に分離**し WebSocket で接続します。
`policy_backend:=local` にすると同一プロセスで openpi を読み込むこともできます。

| コンポーネント | 実体 |
| --- | --- |
| シミュレータ | [hsr-project](https://github.com/hsr-project) の `humble` ブランチ一式（Ignition Gazebo Fortress） |
| ROS 2 ノード | `deploy/hsr_openpi_ros2/hsr_openpi` |
| サービス定義 | `deploy/hsr_openpi_ros2/hsr_openpi_msgs` |
| ポリシーサーバ | `server/serve_hsr_policy_ws.py`（既存）+ `docker/ros2/Dockerfile.server` |
| ベースモデル | [`airoa-org/airoa-pi05-hsr-base`](https://huggingface.co/airoa-org/airoa-pi05-hsr-base) |

---

## 2. セットアップ

### 2.1 HSR シミュレータ用ワークスペースの作成

```bash
# airopi-share と同じ親ディレクトリに hsr_ros2_ws を作る
./scripts/ros2/setup_hsr_ros2_ws.sh
```

`hsr-project` の 22 リポジトリ（`humble` ブランチ）に加えて、arm64 向け deb が
公開されていない `ros_gz` と `gz_ros2_control` をソースで取得します。
実機専用の `hsrb_robot_launch` と、専用 SDK が必要な `tmc_pgr_camera` は
`COLCON_IGNORE` されます。

### 2.2 ベースモデルのダウンロード

```bash
./scripts/ros2/download_base_model.sh
# -> ../checkpoints/airoa-pi05-hsr-base  (約 6.6 GB)
```

チェックポイントには `experiment_config/experiment_config.yaml` が同梱されて
いるので、サーバ側で config 名を指定する必要はありません。

### 2.3 イメージのビルドとコンテナ起動

```bash
cd docker/ros2
./build-sim-image.sh                                   # hsr-ros2-sim:humble
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build openpi-server
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d hsr-sim
./build-workspace.sh                                   # rosdep + colcon build
```

`build-workspace.sh` は既定で「Gazebo + openpi クライアント」に必要な
パッケージだけをビルドします（`--all` で全 86 パッケージ）。

---

## 3. 実行

### 3.1 シミュレータの起動

```bash
docker compose exec hsr-sim bash
ros2 launch hsr_openpi hsr_sim.launch.py                    # ヘッドレス・empty world
# ros2 launch hsr_openpi hsr_sim.launch.py world:=apartment_no_objects \
#     robot_pos_x:=5.0 robot_pos_y:=6.6
# ros2 launch hsr_openpi hsr_sim.launch.py headless:=false  # GUI（DISPLAY が必要）
```

`headless:=true`（既定）では `gz sim -s --headless-rendering` として起動し、
X サーバなしで EGL によるカメラレンダリングを行います。

確認:

```bash
ros2 topic hz /head_rgbd_sensor/rgb/image_rect_color
ros2 topic hz /hand_camera/image_raw
ros2 topic echo /joint_states --once
```

### 3.2 ポリシーサーバの起動

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d openpi-server
docker compose logs -f openpi-server        # "Serving policy ..." が出れば OK
```

### 3.3 推論ノードの起動

```bash
docker compose exec hsr-sim bash
ros2 launch hsr_openpi hsr_openpi.launch.py \
    instruction:="Grasp the bottle." auto_start:=true
```

実行中の指示文の差し替え:

```bash
ros2 service call /hsr_openpi/update_instruction hsr_openpi_msgs/srv/StringTrigger \
    "{message: 'Open the oven toaster'}"
```

開始 / 停止:

```bash
ros2 service call /hsr_openpi/start std_srvs/srv/Trigger
ros2 service call /hsr_openpi/stop  std_srvs/srv/Trigger
```

初期姿勢へ戻す:

```bash
ros2 launch hsr_openpi reset_pose.launch.py
```

---

## 4. トピック対応表（実機 / シミュレータ）

シミュレータと実機でトピック名が異なるため、すべて ROS パラメータ化してあり、
`robot_profile:=sim|real` で切り替えます（`config/{sim,real}_topics.yaml`）。

| 用途 | 実機 (ROS 1 / hsrb bringup) | Ignition Gazebo (hsrb_gazebo_bringup) |
| --- | --- | --- |
| 頭部カメラ | `/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed` | `/head_rgbd_sensor/rgb/image_rect_color` (raw) |
| ハンドカメラ | `/hsrb/hand_camera/image_raw/compressed` | `/hand_camera/image_raw` (raw) |
| 関節角 | `/hsrb/joint_states` | `/joint_states` |
| アーム指令 | `/hsrb/arm_trajectory_controller/command` | `/arm_trajectory_controller/joint_trajectory` |
| ヘッド指令 | `/hsrb/head_trajectory_controller/command` | `/head_trajectory_controller/joint_trajectory` |
| グリッパ指令 | `/hsrb/gripper_controller/command` | `/gripper_controller/joint_trajectory` |
| 台車指令 | `/hsrb/command_velocity` (Twist) | `/omni_base_controller/cmd_vel` (Twist) |
| 把持アクション | `/hsrb/gripper_controller/grasp` | `/gripper_controller/grasp` |

ROS 2 の `joint_trajectory_controller` はトピック名が `~/command` ではなく
`~/joint_trajectory` である点に注意してください。

---

## 5. 主な launch 引数

`ros2 launch hsr_openpi hsr_openpi.launch.py <arg>:=<value>`

| 引数 | 既定値 | 意味 |
| --- | --- | --- |
| `robot_profile` | `sim` | トピック構成（`sim` / `real`） |
| `policy_backend` | `websocket` | `websocket`（別プロセス） / `local`（同一プロセス） |
| `policy_host` / `policy_port` | `127.0.0.1` / `8010` | ポリシーサーバの接続先 |
| `checkpoint_dir` / `config_yaml` | `""` | `policy_backend:=local` のときのみ使用 |
| `instruction` | `Grasp the bottle.` | 初期指示文 |
| `update_freq` | `10` | 推論結果 action chunk のレート [Hz] |
| `adopted_action_chunks` | `10` | 1 推論あたり消費する chunk 数 |
| `upsample` / `upsample_hz` / `upsample_method` | `true` / `100` / `spline` | 実行レートへの補間 |
| `action_smoothing` / `ema_alpha` / `ma_window` | `ema` / `0.2` / `5` | 平滑化 |
| `gripper_mode` | `hybrid` | `continuous` / `discrete` / `hybrid` |
| `policy_image_order` | `bgr` | ポリシーに渡す画像のチャネル順（後述） |
| `require_control_mode` / `auto_start` | `true` / `false` | `/control_mode == auto` を待つか |
| `save_exec_trace` | `false` | 実行トレース（npz + png）の保存 |

---

## 6. 実装メモ

### 6.1 画像のチャネル順（要注意）

学習側とデプロイ側でチャネル順が食い違っています。

**学習側（RGB）** — `deploy/hsr_data_collection/conversion/` の 3 段階を追うと

```python
# rosbag2pkl.py  : 受信 JPEG -> RGB 配列
img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, ::-1]
# rosbag2pkl.py  : RGB 配列を BGR に戻して保存 => 見た目が正しい JPEG
cv2.imwrite(path, image[:, :, ::-1])
# pkl2np.py / pkl2rlds.py : 読み戻して RGB に
image = cv2.imread(impath)[:, :, ::-1]
```

となり、データセットには **RGB** が入ります。

**デプロイ側（BGR）** — ROS 1 の推論ノード
`deploy/hsr_openpi_deploy/scripts/hsr_openpi.py` は

```python
image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, :]    # BGR のまま
```

で **BGR のまま**渡しており、ICRA 評価ランタイム
(`tamukohlaboratory/airoa-evaluation-ICRA`) の ROS 2 クライアントも
`rgb8` を `COLOR_RGB2BGR` で BGR に揃えてから渡しています。

さらに、転送方式によって元のチャネル順も変わります。`CompressedImage` を
`cv2.imdecode` すると **BGR**、Ignition Gazebo が publish する raw 画像は
**`rgb8`** です。そのため本ノードは「条件付きで反転」ではなく、
**`policy_image_order`（既定 `bgr`）へ正規化**します。そうしないと
シミュレータでは RGB、実機では BGR がポリシーに入ってしまいます。

既定を `bgr` にしているのは、公開チェックポイントで実績のある 2 つの
デプロイ実装に合わせる方が安全と判断したためです。学習パイプライン側（RGB）に
合わせたい場合は `policy_image_order:=rgb` にしてください。どちらが良いかは
実機・シミュレータでの A/B で確認することを推奨します。

### 6.2 JAX のバージョン

`pyproject.toml` は `jax[cuda12]==0.5.3` を固定していますが、jaxlib 0.5.3 の
XLA は **sm_121（NVIDIA GB10 / DGX Spark）を認識できず**、float16 / bfloat16 の
カーネル生成時に

```
Unsupported conversion from bf16 to f16
LLVM ERROR: Unsupported rounding mode for conversion.
```

で落ちます（pi0.5 は params を bfloat16 で復元するため即座に踏みます）。
`docker/ros2/Dockerfile.server` は sm_121 で動作する jaxlib を使用します
（`--build-arg JAX_VERSION=...` で変更可能）。x86 + Ampere/Hopper など、
元の 0.5.3 で問題ない環境ではそちらを指定しても構いません。

### 6.3 サーバイメージが軽い理由

`docker/ros2/Dockerfile.server` は lerobot / torch / torchcodec / ffmpeg 7 を
**入れていません**。公開されている HSR チェックポイントは JAX/orbax 形式で、
`server/serve_hsr_policy_ws.py` は LeRobot のデータパイプラインを import しない
ためです。これにより aarch64 で PyAV を FFmpeg 7 に対してソースビルドする
手間（と失敗）を回避しています。学習を行う場合は従来どおり
`docker/Dockerfile.train*` を使ってください。

### 6.4 arm64 でのシミュレータ

* `packages.ros.org` の ignition ミラーは arm64 で依存関係が壊れている
  （`ignition-fortress` が要求する `libignition-sensors6 >= 6.8.1` が無い）ため、
  `packages.osrfoundation.org` を apt pin 付きで追加しています。
* `ros-humble-ros-gz-sim` と `ros-humble-gz-ros2-control` は arm64 deb が
  存在しないため、ワークスペース内の `src/_extern` でソースビルドします。

---

## 7. ポリシーサーバなしでの動作確認

GPU やチェックポイントが無くても、モックサーバで ROS 2 側の配線を確認できます。

```bash
# 端末 1（hsr-sim コンテナ内）
python3 /home/hsr/airopi-share/deploy/hsr_openpi_ros2/tools/mock_policy_server.py \
    --port 8010 --pattern wiggle

# 端末 2
ros2 launch hsr_openpi hsr_openpi.launch.py auto_start:=true
```

アーム・ヘッド・台車が正弦波状に動けば、観測取得 → 推論 → 補間 → 平滑化 →
コントローラ送信までの経路は正常です。

---

## 8. トラブルシューティング

| 症状 | 対処 |
| --- | --- |
| `Waiting for observations: head_rgb(...)` が出続ける | Gazebo のセンサが動いているか `ros2 topic hz` で確認。ヘッドレス時は EGL が必要（`NVIDIA_DRIVER_CAPABILITIES=all`） |
| `Action NOT executed (control mode is not active)` | `auto_start:=true` にするか `/hsr_openpi/start` を呼ぶ |
| ノードが `/clock` 待ちで止まる | Gazebo 未起動、または `use_sim_time:=false` にする |
| ポリシーサーバに繋がらない | ポート衝突（既定 8010）。`POLICY_SERVER_PORT` と `policy_port` を合わせる |
| `Unsupported rounding mode for conversion` | §6.2 参照。JAX のバージョンを上げる |
| メモリ不足で落ちる | GB10 などの unified memory 機では他プロセス（vLLM 等）の GPU メモリを解放する |
