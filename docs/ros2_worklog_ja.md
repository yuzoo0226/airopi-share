# ROS 2 版 HSR + Gazebo 推論環境: 構築記録と既知の落とし穴

このドキュメントは「別の PC で同じ環境を再現する」ための実作業記録です。
手順そのものは [`ros2_deploy_ja.md`](ros2_deploy_ja.md) にまとめてあり、
ここには **実際に何をしたか / 何が失敗したか** を残します。

---

## 1. 検証した環境

| 項目 | 値 |
| --- | --- |
| マシン | NVIDIA DGX Spark 相当（GB10 Grace Blackwell、CPU/GPU で 128 GB のユニファイドメモリ） |
| アーキテクチャ | **aarch64 (arm64)** |
| OS | Ubuntu 24.04.4 LTS (noble) |
| NVIDIA ドライバ | 580.126.09 / CUDA 13.0 |
| GPU compute capability | **sm_121 (12.1)** |
| Docker | 29.1.3 + NVIDIA Container Toolkit (`runtime: nvidia`) |
| ホスト側 ROS | なし（すべてコンテナ内） |

> 重要: ホストは Ubuntu 24.04 なので ROS 2 Humble はネイティブに入りません。
> Humble / Gazebo Fortress はすべて `hsr-ros2-sim:humble` コンテナの中です。
> x86_64 + Ubuntu 22.04 のマシンならもっと素直に構築できます（後述の
> arm64 固有の回避策の多くが不要になります）。

## 2. 実際に踏んだ手順

```bash
# 1) リポジトリ（fork）
git clone https://github.com/yuzoo0226/airopi-share.git
cd airopi-share && git checkout -b devel/ros2

# 2) HSR シミュレータのワークスペース（hsr-project の humble ブランチ 22 リポジトリ）
./scripts/ros2/setup_hsr_ros2_ws.sh

# 3) ベースモデル (6.2 GB)
./scripts/ros2/download_base_model.sh airoa-org/airoa-pi05-hsr-base

# 4) イメージ
cd docker/ros2
./build-sim-image.sh                                   # ROS 2 Humble + Fortress
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build openpi-server

# 5) ワークスペースのビルド（コンテナ内、67 + 3 パッケージ / 約 7 分）
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d hsr-sim
./build-workspace.sh

# 6) 実行
docker compose exec hsr-sim bash -lc \
  'ros2 launch hsr_openpi hsr_sim.launch.py world:=apartment robot_pos_x:=5.0 robot_pos_y:=6.6'
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d openpi-server
docker compose exec hsr-sim bash -lc \
  'ros2 launch hsr_openpi hsr_openpi.launch.py auto_start:=true instruction:="Grasp the bottle."'
```

## 3. 動作確認できた数値

| 項目 | 値 |
| --- | --- |
| チェックポイント読み込み | 7.8 GiB を約 3 秒（2.5 GiB/s） |
| pi0.5 推論レイテンシ | **約 340 ms / chunk**（GB10、JAX 0.6.2 GPU、bf16） |
| 実行レート | 50 Hz（`update_freq=10`, `upsample_hz=50`） |
| 1 推論あたりの行動長 | 10 chunk = 1.0 s 分（`adopted_action_chunks=10`） |
| head カメラ | 640x480 `rgb8`、empty world で 12〜14 Hz、apartment world で約 39 Hz |
| hand カメラ | 640x480 `rgb8`、6〜7 Hz |
| `/joint_states` | 約 195 Hz |

推論 340 ms に対して 1 回の推論が 1.0 秒分の行動を出すので、
チャンク間のギャップは発生せず連続動作します。

---

## 4. ハマった点と対処（失敗の記録）

### 4.1 シミュレータ側 (arm64)

**(a) `ignition-fortress` が arm64 で依存関係を解決できない**

```
ignition-fortress : Depends: libignition-sensors6-camera (>= 6.8.1)
                    but 6.8.0-1~jammy is to be installed
```

`packages.ros.org` にミラーされている ignition パッケージ群は arm64 で
バージョンが揃っていません。`packages.osrfoundation.org/gazebo/ubuntu-stable`
を追加し、apt pin（Pin-Priority: 1001）で必ず OSRF 側を使うようにして解決。

**(b) `ros-humble-ros-gz-sim` / `ros-humble-gz-ros2-control` の arm64 deb が無い**

`ros-humble-ros-gz-bridge` と `-image` は存在するのに `-sim` は存在しません。
`gz_ros2_control` も humble 向けには published されていません（iron / rolling のみ）。
→ ワークスペースの `src/_extern/` に `gazebosim/ros_gz`（humble）と
`ros-controls/gz_ros2_control`（humble）をソースで置いて colcon でビルド。

**(c) `rosdep` が `hsrb_bringup` を解決できない**

`hsrb_launch/hsrb_robot_launch` は実機専用で、公開されていない
`hsrb_bringup` に依存しています。hsr-project の公式手順でも
`rm -rf hsrb_launch/hsrb_robot_launch` するよう書かれています。
本リポジトリでは削除ではなく `COLCON_IGNORE` を置き、
rosdep には `--skip-keys hsrb_bringup` を渡しています。
同じ理由で `tmc_drivers/tmc_pgr_camera`（PointGrey の SDK が必要）も除外。

**(d) `ros_gz_point_cloud` が `roscpp` / `catkin` を要求する**

`ros_gz` リポジトリには ROS 1 用のパッケージが同梱されています。
`COLCON_IGNORE` を置いただけでは **rosdep は package.xml を見てしまう**ため、
`build-sim-image.sh` の manifest 収集側で COLCON_IGNORE 配下を除外するよう
修正しました（イメージビルド時に rosdep が落ちる）。

**(e) rosdep のキャッシュがユーザ単位**

イメージビルド中に root で `rosdep update` しても、コンテナは uid 1000 の
`hsr` ユーザで動くため実行時に

```
ERROR: your rosdep installation has not been initialized yet.
```

となります。Dockerfile 内で `USER hsr` に切り替えてもう一度 `rosdep update`
を実行して解決。

**(f) `hsrb_gazebo_bringup` にヘッドレス起動のオプションが無い**

`gazebo_bringup.launch.py` は `gz_args` を
`' -r -v 3 ' + world_file_name` と決め打ちで組み立てており、
`-s`（サーバのみ）や `--headless-rendering` を渡す手段がありません。
X サーバ無しの環境では GUI が立ち上がらず起動に失敗します。
→ `world_file_name` がそのまま連結される点を利用して
`world_file_name:="--headless-rendering -s <world>"` と渡す
ラッパー launch (`hsr_openpi/hsr_sim.launch.py`) を用意しました。
`--headless-rendering` によりカメラは EGL 経由で GPU レンダリングされます
（`NVIDIA_DRIVER_CAPABILITIES=all` が必要）。

**(g) ビルド時のメモリ**

GB10 は 128 GB を CPU と GPU で共有します。他プロセス（vLLM）が
GPU メモリを確保していると colcon の並列ビルドが OOM します。
`build-workspace.sh` は `--parallel-workers 3` / `MAKEFLAGS=-j3` を既定にし、
`COLCON_WORKERS` / `MAKE_JOBS` で上書きできるようにしています。

### 4.2 推論サーバ側 (JAX / sm_121)

**(h) jaxlib 0.5.3 が sm_121 を認識できない**

`pyproject.toml` の `jax[cuda12]==0.5.3` は GPU 自体は認識する
（`CudaDevice(id=0)`, `compute_capability 12.1`）のに、half precision の
カーネル生成で落ちます。

```python
jnp.ones((256,256), jnp.float32) @ ...   # OK
jnp.ones((256,256), jnp.float16) @ ...   # LLVM ERROR
```
```
Unsupported conversion from f16 to f16
LLVM ERROR: Unsupported rounding mode for conversion.
```

pi0.5 は params を bfloat16 で復元するため即座に踏みます。
検証結果:

| jax | numpy 要件 | sm_121 f32 | f16 | bf16 |
| --- | --- | --- | --- | --- |
| 0.5.3 | >=1.22 | OK | **NG** | **NG** |
| 0.6.2 | >=1.26 | OK | OK | OK |
| 0.7.0 | >=1.26 | OK | OK | OK |
| 0.7.2 | **>=2.0** | OK | OK | OK |

→ numpy 1.x を保てる **0.6.2** を採用。

**(i) `orbax-checkpoint==0.11.13` が jax 0.6 で import できない**

```
AttributeError: module 'jax.experimental.layout' has no attribute 'DeviceLocalLayout'
```

jax 0.6 でこのクラスが改名されました。0.11.14 以降で解消。

**(j) `chex` が jax を勝手に 0.7.1 へ上げる**

依存解決の途中で jax が上がり、(i) が再発しました。
`/etc/pip-constraints.txt` に `numpy<2 / jax==X / jaxlib==X / orbax==Y / flax==Z`
を書き、`PIP_CONSTRAINT` でイメージ全体に効かせて固定しています。
同じ理由で numpy も 2.4.6 に上がっていました。

**(k) JAX しか使わないのに torch と pytest が必要**

`src/openpi/models/model.py` が無条件に `import torch` し、
その先の `openpi/models_pytorch/gemma_pytorch.py` が `import pytest` します。
aarch64 の `torch==2.7.1` wheel は約 100 MB の CPU ビルドで、CUDA ランタイムを
引きずらないため追加しても軽量です。

**(l) ffmpeg / lerobot は入れない**

学習用イメージは PyAV を FFmpeg 7 に対してソースビルドしますが、
**推論だけなら不要**です。公開チェックポイントは JAX/orbax 形式で、
`server/serve_hsr_policy_ws.py` は LeRobot のデータパイプラインを
import しません（`openpi.training.config` は lerobot を import しない。
lerobot を使うのは `data_loader.py` / `fast_lerobot_*.py` のみ）。
これで aarch64 での ffmpeg 沼を丸ごと回避しています。

### 4.3 実行環境まわり

**(m) ポート 8000 の衝突**

既存の vLLM が `127.0.0.1:8000` を使っていたため、
`network_mode: host` のコンテナが bind できず

```
OSError: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000)
```

→ 既定を **8010** に変更（`POLICY_SERVER_PORT` / `policy_port`）。

**(n) ユニファイドメモリの取り合い**

GB10 では GPU メモリ = システムメモリです。vLLM（GPT-OSS-Swallow-120B,
`gpu_memory_utilization=0.85`）が 105.8 GB を確保していて空きが 2〜6 GB しか
なく、pi0.5 が載りませんでした。`docker stop gpt-agent-vllm` で 112 GB 確保。
**別 PC で動かす場合も、GPU メモリを 15〜25 GB 空けてから**起動してください。

**(o) `pid: host` は危険**

compose に `pid: host` を書いていたため、コンテナ内で実行した `pkill` が
**ホスト側のプロセスまで巻き込みました**。compose から削除済みです。

**(p) `pkill -f` が自分自身にマッチする**

`pkill -f "ros2 launch"` はそのコマンドを実行しているシェル自身の
コマンドラインにもマッチして自殺します。`docker/ros2/stop-sim.sh` では
`'ros2 laun[c]h'` のようにブラケットで回避しています。

**(q) 画像のチャネル順**

`CompressedImage` は `cv2.imdecode` で **BGR**、Gazebo の raw 画像は **`rgb8`**
です。単純な「反転するか否か」のフラグだと、シミュレータでは RGB、実機では
BGR がポリシーに入る、という取り違えが起きます。
`policy_image_order`（既定 `bgr`）へ正規化する実装に変更しました。
詳細は [`ros2_deploy_ja.md`](ros2_deploy_ja.md) の 6.1。

### 4.4 データ収集 / 変換 / 学習

**(r) `ros2 bag record` を kill すると bag が壊れる**

rosbag2 は **SIGINT を受けたときにだけ** `metadata.yaml` を書きます。
`kill -9` すると

```
Could not find metadata in bag directory .../test01
```

となり `ros2 bag info` も変換スクリプトも開けません。
`ros2 bag reindex <dir> -s mcap` で復旧できます
（`no message indices found, falling back to reading in file order` という
警告は出ますが読めるようになります）。
`collect_data.launch.py` は `hsr_random_motion` の終了を `OnProcessExit` で
拾って launch 全体を `Shutdown` させ、レコーダに SIGINT が届くようにしています。

**(s) `uv run` が環境を巻き戻す**

`uv run <cmd>`（`--no-sync` なし）は実行前に環境を `uv.lock` に同期し直します。
その結果、

* GB200 オーバーレイで入れた `torch 2.10.0+cu128` が `2.7.1+cpu` に戻る
* あとから `uv pip install` したパッケージ（`rosbags` など）が消える

という事故が起きます（実際に `Uninstalled 18 packages / Installed 18 packages`
が出ました）。`${UV_PROJECT_ENVIRONMENT}/bin/python` を直接呼ぶか
`uv run --no-sync` を使ってください。

**(t) `uv pip install` がプロジェクトの override を引きずる**

プロジェクトディレクトリ内で実行した `uv pip install` は
`pyproject.toml` の `[tool.uv] override-dependencies`
（`ml-dtypes==0.4.1`, `tensorstore==0.1.74`）を適用します。
jax 0.6.2 は `ml_dtypes>=0.5` を要求するため、プロジェクト内で入れると

```
AttributeError: module 'ml_dtypes' has no attribute 'float8_e3m4'
```

で jax が import できません。`cd /` してから入れる必要があります
（`scripts/ros2/apply_sm121_jax_overlay.sh` がこれを行います）。
なお jax 0.6.2 を入れると numpy が 2.x に上がるので、最後に `numpy<2` を
入れ直します。

**(u) 学習側の jax も sm_121 対応版が必要**

推論サーバと同じ理由で、学習コンテナの `uv.lock` 由来の jax 0.5.3 も
sm_121 では bf16 を扱えません。`scripts/ros2/apply_sm121_jax_overlay.sh` で
0.6.2 + orbax 0.11.14 に上げます。

**(v) LeRobot 0.1.0 の `save_episode()` に `task` 引数が無い**

openpi が pin している LeRobot revision では、自然言語タスクは
**フレームごと**に `add_frame({... , "task": "..."})` として渡し、
`save_episode()` は引数なしで呼びます。
`save_episode(task=...)` と書くと `TypeError` になります。

**(w) 学習コンテナに checkpoints ディレクトリがマウントされていない**

`docker/docker-compose.yml` はリポジトリとデータセットしかマウントしないため、
`weight_loader.params_path` にローカルのベースモデルを指定できません。
`docker/ros2/docker-compose.train.yml`（override）で `/home/checkpoints` と
`/home/bags` を足しています。

**(x) ハンドカメラのレンダリングが遅い**

Ignition が HSR の全カメラ（頭部 RGBD・ステレオ 2 台・head_center・ハンド）を
レンダリングするため、ハンドカメラは 6〜9 Hz しか出ません。
10 fps のデータセットでは同じ画像が複数フレームで使われることがあります。
不要なカメラを URDF から外すのが根本対処です。

### 4.5 学習設定

**(y) `image_encoder_mode` を公開チェックポイントに合わせないと視覚重みが落ちる**

pi0 は画像エンコーダのモジュール名を次のように決めます
(`src/openpi/models/pi0.py::_image_encoder_module_name`)。

| `image_encoder_mode` | モジュール名 |
| --- | --- |
| `shared` (既定) | 全ての画像キー → `img` |
| `per_image` | `IMAGE_KEYS[0]` → `img`、それ以外 → `img_<key>` |

`airoa-pi05-hsr-base` は **`per_image`** で学習されています。自前の YAML で
`image_encoder_mode` を書き忘れると既定の `shared` になり、
`CheckpointWeightLoader` は名前が一致する `PaliGemma/img/...` だけを読み込みます。
HSR の場合 `IMAGE_KEYS[0]` は **ゼロ埋めしてマスクした `base_0_rgb`** のスロットなので、

* 実際に使われる hand / head 用の HSR 適応済みエンコーダ
  (`img_left_wrist_0_rgb`, `img_right_wrist_0_rgb`) は**捨てられ**、
* ほぼ素の PaliGemma 重みだけが載る

という状態になります。エラーも警告も出ません。

見分け方は step 0 の `param_norm` です。

```
shared    : param_norm=2019.61
per_image : param_norm=2228.31
```

`configs/experiments/example_hsr_*.yaml` では `model.image_encoder_mode:
per_image` を明示しています。**推論側は影響を受けません**
（サーバはチェックポイント同梱の `experiment_config.yaml` を自動検出するため）。

---

## 5. 別 PC へ移すときのチェックリスト

1. **アーキテクチャ**: x86_64 なら 4.1 (a)(b) の回避策は不要になる可能性が高い
   （`ros-humble-ros-gz-sim` / `gz-ros2-control` の deb が存在するため）。
   その場合でもソースビルドのままで動きます。
2. **GPU**: sm_121 以外（Ampere / Ada / Hopper）なら
   `--build-arg JAX_VERSION=0.5.3` に戻して pyproject 準拠にできます。
   まず `docker run --rm --runtime=nvidia airopi-server:arm64 python -c
   "import jax,jax.numpy as jnp; print(jnp.ones((256,256),jnp.bfloat16)@jnp.ones((256,256),jnp.bfloat16))"`
   で bf16 が通るか確認してください。
3. **GPU メモリ**: 15〜25 GB 以上空けておく。
4. **ポート**: `POLICY_SERVER_PORT` と launch の `policy_port` を合わせる。
5. **X サーバ**: 無くても `headless:=true`（既定）で動きます。
   GUI を出す場合は `DISPLAY` を通して `headless:=false`。
6. **実機に向ける場合**: `robot_profile:=real` と `policy_host` を指定。
   カメラトピック名は HSR の個体差があります（`/head_rgbd_sensor/rgb/image_rect_color/compressed`,
   `/head_rgbd_sensor/rgb/image_raw/compressed`, `/head_rgbd_sensor/color/image_raw/compressed` など）。
   `config/real_topics.yaml` を実機に合わせて調整してください。

---

## 6. Gazebo が exit 134 で落ちる件（2026-08-25 追記）

長時間の収集中に Ignition が `exit code 134`（SIGABRT）で落ちる。1 チャンク
100 エピソードのうち 34〜94 本で発生し、再現条件は特定できていないが、
**原因は ODE ではなく台車コントローラ**であることまで判明した。

abort の直前に出ているのはこの 3 行:

```
[omni_base_controller] Too big joint velocity! [right, left, steer]=[2.37e+48, 0.0, 0.0]
[controller_manager] The update call of the following controller returned an error: 'omni_base_controller'
ODE INTERNAL ERROR 1: assertion "aabbBound >= dMinIntExact && aabbBound < dMaxIntExact" failed in collide() [collision_space.cpp:460]
```

`hsrb_base_controllers::OmniBaseController` が右車輪に 2.37×10^48 という
値を出し、ロボットが事実上無限遠へ飛ぶ。その結果 AABB が ODE のハッシュ空間の
整数範囲を超え、`dxHashSpace::collide` のアサーションが `dDebug` → `abort` を
呼ぶ。つまり **ODE のアサーションは症状であって原因ではない**。
`<collision_detector>bullet` に替えても直らない（そちらは DART の
`BoxedLcpConstraintSolver.cpp:229` で即死する）のはこのため。

観測できた事実:

* この `Too big joint velocity!` は落ちる直前の **1 回だけ**出る。
  常時出ている警告が積み重なって落ちるのではない。
* 発生時刻はエピソード 51 の終了 1.0 秒後で、`RESET` で
  `world.set_pose("hsrb", ...)` がロボットをテレポートさせる瞬間と一致する。
  ただし別チャンクではエピソード途中でも落ちているため、テレポートが
  唯一のトリガーとは言い切れない。
* OOM ではない（`dmesg` に記録なし、空き 90 GB、Gazebo の RSS は 1 GB 未満）。
* `motion_command_limitter_controller` は台車の車輪インタフェースを
  握っているが、この値を止めていない。

### 対処

根治は TMC 側のコントローラの問題なので、**落ちる前提で運用**している。
`scripts/ros2/collect_pick_chunks.sh` がチャンクごとに Gazebo を立て直し、
recorder が SIGINT を受け取れずに壊れた bag は `ros2 bag reindex` →
`ros2 bag convert` で修復する。落ちてもそのチャンクまでのデータは失われない
（例: chunk 0 は 94 本すべて回収できている）。

### 次に落ちたときに原因を追うには

Gazebo の出力は `/tmp/gz.log` にリダイレクトされ、次チャンクの開始時に
切り詰められるため、abort 行は放っておくと毎回消える。プロセスの消滅を
検知してログを退避すること。上記のスタックトレースはそうやって採取した。
