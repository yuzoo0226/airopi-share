# AiroPi for HSR — 統合ガイド (日本語版)

π0 / π05 の HSR 向け fine-tuning・評価・デプロイを扱うリポジトリです。
以下の 2 系統の実行環境を 1 本のコードベースで扱えることを目的としています。

- **ローカル GPU マシン (Docker)** — 再現性のある学習環境 (非 Blackwell / Blackwell GB200 / deploy の 3 モード)
- **HPC (SLURM / PBS+qsub)** — multi-node での本番学習

スタックとしては **pixi + uv + Docker** の層構成で、ローカル学習は Docker 経由に
統一しています。ホストで直接 `pixi run uv run scripts/train.py ...` を叩く
「pixi + uv 単体」の学習フローは現状サポート外です (pixi は HPC launcher 内で
自動的に使われます)。

学習設定は **YAML (`configs/experiments/*.yaml`) を推奨** します。tyro ベースの
`CONFIG_NAME` 経路は互換のため残っていますが、マルチノードも Docker も
Blackwell overlay もすべて YAML を基準に書かれています。

> English: See [`README.md`](README.md) for the English comprehensive guide.
> PyTorch 学習もコードは存在しますが、本リポジトリの HSR 学習では性能面から
> **JAX 学習を推奨** します。

---

## 目次

1. [概要と対応マトリクス](#1-概要と対応マトリクス)
2. [リポジトリ構成ダイジェスト](#2-リポジトリ構成ダイジェスト)
3. [クイックスタート (TL;DR)](#3-クイックスタート-tldr)
4. [事前準備](#4-事前準備)
5. [学習設定 (YAML 推奨)](#5-学習設定-yaml-推奨)
6. [ローカル環境](#6-ローカル環境)
7. [モデルのデプロイ](#7-モデルのデプロイ)
8. [HPC — SLURM](#8-hpc--slurm)
9. [HPC — PBS (qsub)](#9-hpc--pbs-qsub)
10. [ワークフロー横断 Tips](#10-ワークフロー横断-tips)
11. [.env / .env.local 変数早見表](#11-env--envlocal-変数早見表)
12. [トラブルシューティング](#12-トラブルシューティング)
13. [付録](#13-付録)

---

## 1. 概要と対応マトリクス

| ユースケース | 推奨スクリプト | 設定方式 | 備考 |
| --- | --- | --- | --- |
| ローカル Docker (非 Blackwell 学習) | `BUILD-DOCKER-CONTAINER.sh train` + `RUN-DOCKER-CONTAINER.sh train` | YAML | CUDA 12.2 ベース、**ローカル学習の標準経路** |
| ローカル Docker (Blackwell GB200 学習) | `BUILD-DOCKER-CONTAINER.sh train-gb200` + `RUN-DOCKER-CONTAINER.sh train-gb200` | YAML | cu128 torch を overlay |
| ローカル Docker (HSR デプロイ) | `BUILD-DOCKER-CONTAINER.sh deploy` + `RUN-DOCKER-CONTAINER.sh deploy` | — | ROS Noetic 内包 |
| policy server コンテナ | `docker build -f server/Dockerfile -t openpi-serve .` | YAML / tyro | 評価パイプライン用 |
| HPC SLURM 単ノード | `RUN-UV-SBATCH.sh` / `RUN-UV-SBATCH-8GPU.sh` | tyro (`CONFIG_NAME`) | スクリプト内部で pixi+uv を使用 |
| HPC SLURM multi-node | `RUN-UV-SBATCH-OPENPI-MULTINODE.sh` | YAML | 8 GPU × N ノード想定 |
| HPC PBS 単ノード | `RUN-UV.sh` | YAML / tyro | スクリプト内部で pixi+uv を使用 |
| HPC PBS multi-node (推奨) | `RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh` | YAML | HPCX + mpirun 直起動 |
| HPC PBS multi-node (高機能版) | `RUN-UV-QSUB-OPENPI-MULTINODE.sh` | YAML | UCX チューニング可 |
| HPC Slurm 対話ノード取得 | `./run_interactive_node.sh` | — | `salloc` ラッパ。取得後は `sbatch` 系launcherを使用 |
| HPC Singularity 単ノード | `BUILD_SINGULARITY.sh` + `RUN-SINGULARITY*.sh` | tyro / YAML | Docker 不可環境用 |

> **ローカルでの単体 `pixi run uv run scripts/train.py ...` 実行は現状サポート外です。**
> ローカル学習を回す場合は必ず Docker (`train` / `train-gb200`) を使ってください。

---

## 2. リポジトリ構成ダイジェスト

```
AiroPi/
├── README.md                        # 英語の統合ガイド
├── README_ja.md                     # 本ファイル (日本語の統合ガイド)
│
├── pixi.toml, pixi.lock             # システム依存 (ffmpeg 7 / pkg-config / compilers)
├── pyproject.toml, uv.lock          # Python 依存関係 (Python 3.11)
│
├── .env.example                     # 実験設定テンプレ  (コミット対象)
├── .env.local.example               # 個人設定テンプレ  (コミット対象)
│                                    # 実体の .env / .env.local は gitignore
│
├── BUILD-DOCKER-CONTAINER.sh        # docker compose build
├── RUN-DOCKER-CONTAINER.sh          # docker compose up + exec bash
├── RUN-DOCKER-COMMON.sh             # train / train-gb200 / deploy 共通ロジック
├── RUN-DOCKER-TRAIN.sh              # コンテナ内学習ヘルパ
│
├── RUN-UV-COMMON.sh                 # pixi + uv 系 launcher の共通ロジック
├── RUN-UV.sh                        # PBS 単ノード (YAML / tyro 両対応)
├── RUN-UV-SBATCH.sh                 # SLURM 単ノード 1 GPU (tyro)
├── RUN-UV-SBATCH-8GPU.sh            # SLURM 単ノード 8 GPU (tyro)
├── RUN-UV-SBATCH-OPENPI-MULTINODE.sh     # SLURM multi-node (YAML)
├── RUN-UV-QSUB-OPENPI-MULTINODE.sh       # PBS multi-node (UCX チューニング版)
├── RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh# PBS multi-node (SIMPLE, 推奨)
├── RUN-UV-EVAL.sh, RUN-UV-DATA-CHECK.sh, ...
├── BUILD_SINGULARITY.sh, RUN-SINGULARITY*.sh  # Singularity 経路
├── run_interactive_node.sh          # Slurm 対話ノード取得 (salloc ラッパ)
│
├── docker/
│   ├── docker-compose.yml           # モード共通 compose
│   ├── Dockerfile.train             # 非 Blackwell 学習 (CUDA 12.2)
│   ├── Dockerfile.train.gb200       # Blackwell (cu122 + cu128 overlay)
│   ├── Dockerfile.deploy            # HSR デプロイ (ROS Noetic)
│   ├── openpi.def                   # Singularity 定義
│   └── scripts/                     # init, GB200 overlay, ROS 初期化など
│
├── server/
│   ├── Dockerfile                   # policy server (軽量 runtime image)
│   ├── entrypoint.sh
│   └── serve_hsr_policy_ws.py       # WebSocket policy server 本体
│
├── scripts/
│   ├── train.py                     # 学習メイン
│   ├── compute_norm_stats.py        # 正規化統計 (tyro)
│   ├── aggregate_stats_simple.py / aggregate_stats_fast.py  # 並列版など
│   ├── eval_val_loss.py
│   ├── serve_policy.py              # サーバ起動の別エントリ (参考)
│   └── openpi_utils/                # SLURM/PBS 分散ランチャ一式
│
├── configs/experiments/             # 学習設定 (YAML)
│   ├── example_base.yaml            # tracked テンプレ
│   ├── example_lora.yaml            # tracked テンプレ
│   ├── example_multinode.yaml       # tracked テンプレ
│   └── (その他 *.yaml は gitignore でローカル専用)
│
├── src/openpi/                      # 本体コード
├── packages/openpi-client/          # クライアント SDK (サブパッケージ)
└── deploy/                          # HSR ROS 連携資産
```

> `configs/experiments/*.yaml` は `.gitignore` で `example_*.yaml` のみ commit
> 対象です。個別実験 YAML (`pi05_hsr_task6891011_*.yaml` 等) はローカル参考や
> 旧履歴のみで、配布物には含めない運用です。

---

## 3. クイックスタート (TL;DR)

### 3.1 ローカル Docker で学習 (非 Blackwell)

```bash
# ホスト側
export DEEP_PROJECT_NAME=mytest
export DEEP_DATASET_PATH=/path/to/datasets

./BUILD-DOCKER-CONTAINER.sh train
./RUN-DOCKER-CONTAINER.sh train
# → コンテナに exec で入るので、以降はコンテナ内 (`/home/openpi`) で作業:

# 1) norm_stats を計算して YAML の assets_dir/asset_id 配下に置く
#    (未作成のまま学習すると
#    "Normalization stats not found. Make sure to run scripts/compute_norm_stats.py ..."
#    で落ちます)
JAX_PLATFORMS=cpu uv run python scripts/aggregate_stats_fast.py \
  --episodes-stats "${DATA_DIR}/meta/episodes_stats.jsonl" \
  --output-file "assets/example_base/${REPO_ID}/norm_stats.json" \
  --chunk-dir "${DATA_DIR}/data/" \
  --action-column "action.relative" \
  --action-mode "relative"

# 2) 学習 (YAML モード)
uv run scripts/train.py \
  --config-yaml configs/experiments/example_base.yaml \
  --exp-name my_exp --overwrite
```

- `DATA_DIR` / `REPO_ID` は YAML の `dataset.data_dir` / `dataset.repo_id` と
  合わせてください (コンテナ内の `/home/datasets/...` 配下が典型)
- 出力先 `assets/<assets_dir 名>/<asset_id>/norm_stats.json` は
  `configs/experiments/example_base.yaml` の `assets_dir` (既定 `./assets/example_base`)
  と `asset_id` (= `repo_id`) の組み合わせで決まります。自作 YAML では両フィールドに
  合わせて読み替えてください
- 初回の `uv sync` とキャッシュ作成はコンテナ起動時の init スクリプトが自動で
  行うので、ホスト側で `pixi run sync` を叩く必要はありません

### 3.2 ローカル Docker で学習 (Blackwell / GB200)

```bash
export DEEP_PROJECT_NAME=gb200test
export DEEP_DATASET_PATH=/path/to/datasets

./BUILD-DOCKER-CONTAINER.sh train-gb200
./RUN-DOCKER-CONTAINER.sh train-gb200
# cu128 torch overlay がコンテナ内で自動適用される

# norm_stats 計算 → 学習 (§3.1 と同じ手順)
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

### 3.3 HPC PBS で multi-node 学習 (推奨: SIMPLE)

```bash
cp .env.example .env              # 実験設定 (YAML パス, EXP_NAME, etc.)
cp .env.local.example .env.local  # 個人設定 (WANDB_API_KEY, CACHE_ROOT, etc.)
# 両方を編集

qsub RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh
```

---

## 4. 事前準備

### 4.1 リポジトリ取得

LeRobot サブモジュールと Git LFS を含むため、初回は再帰 clone を推奨します。

```bash
git clone --recurse-submodules git@github.com:airoa-org/AiroPi.git
cd AiroPi
# すでに clone 済みなら:
git submodule update --init --recursive
```

### 4.2 `.env` と `.env.local` のコピー

`.env` と `.env.local` の **2 層構成** です。

- `.env` — 実験設定 (YAML パス、EXP_NAME、ノード数、resume/overwrite 等)。
  チーム内で共有しやすい内容を置く
- `.env.local` — 個人/マシン固有 (WANDB API key、CACHE_ROOT、CHECKPOINT_DIR、
  HF_LEROBOT_HOME 等)。個人ごとに異なる内容を置く

両方のテンプレ (`.env.example` / `.env.local.example`) は commit 対象で、実体の
`.env` / `.env.local` は `.gitignore` 済みです。

```bash
cp .env.example .env
cp .env.local.example .env.local
# 両方を編集
```

**読み込み優先順序** (launcher ごとに挙動が違うので注意):

| launcher 系統 | 挙動 | 備考 |
| --- | --- | --- |
| **Family A**: `RUN-UV-COMMON.sh::openpi_load_env_file` を使う全 launcher + `RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh` | `.env` → `.env.local` の順に両方を読み、`.env.local` が後勝ち | `HSR_OPENPI_ENV_FILE` で `.env` のパスを差し替え可 |
| **Family B**: `RUN-SINGULARITY.sh` / `RUN-SINGULARITY-COMPUTE-SIMPLE-NORM.sh` / `download_base_model.sh` / `download_training_weights.sh` | `HSR_OPENPI_ENV_FILE` > `.env.local` > `.env` の **いずれか 1 つだけ** 読む | Family A とは意味論が違う |
| **Family B 内の例外**: `RUN-SINGULARITY-FT.sh` | Family B と同形だが、差し替え変数名が `AIROPI_ENV_FILE` | 他の Singularity 系と不一致 |


最低限設定すべき変数 (詳細は §11 参照):

- `.env`: `OPENPI_CONFIG_YAML`, `EXP_NAME`, `OPENPI_NUM_NODES`, `OPENPI_GRES`,
  `OPENPI_RESUME` / `OPENPI_OVERWRITE`
- `.env.local`: `WANDB_API_KEY`, `CACHE_ROOT`, `CHECKPOINT_DIR`, `HF_LEROBOT_HOME`

### 4.3 pixi + uv の位置づけ

本リポジトリのスタックは **pixi + uv + Docker** の層構成です。

- **Docker (ローカル学習の標準経路)**: コンテナ内で bare `uv` を使います。
  ffmpeg 7 やコンパイラはイメージに内蔵済みで、pixi は不要です
- **HPC (SLURM / PBS)**: `RUN-UV-*.sh` 系 launcher がログインノード / 計算ノード上で
  `pixi run uv sync` / `pixi run uv run ...` を自動的に実行します。
  そのためログインノードに pixi と uv がインストールされている必要があります
- **ホスト直実行 (学習)**: 現在サポート外。ローカルで学習を回す場合は Docker
  を使ってください

HPC クラスタで新規に環境を用意する場合のみ、以下を入れます。

- [pixi インストール手順](https://pixi.sh/latest/#installation)
- [uv インストール手順](https://docs.astral.sh/uv/getting-started/installation/)

> `pixi run sync` / `pixi run sync-blackwell` というコマンドは `pixi.toml` に定義
> されており、ソース閲覧や IDE 用に venv を作る目的で手動で叩く分には問題ありません。
> ただし学習の正式な経路は §6 の Docker 手順に従ってください。

### 4.4 データセット・ベースモデル

- データセットは LeRobot 形式 (`${DATA_DIR}/<repo_id>/{data,meta,videos,...}`) を想定
- `meta/episodes_stats.jsonl` が正規化統計計算に必要
- ベースモデルは YAML の `checkpoints.base_model.<model_type>` (例:
  `gs://openpi-assets/checkpoints/pi05_base/params`) から取得
- オフライン環境では事前に `download_base_model.sh` や `maybe_download(...)` で
  `OPENPI_DATA_HOME` 配下へキャッシュしておくと、multi-node 起動時のダウンロード
  競合を避けられます
  (`RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh` の `Pre-download base model weights` 参照)

---

## 5. 学習設定 (YAML 推奨)

### 5.1 `.env` / `.env.local` と YAML の責務分離

```
.env                  実験・ランチャ設定
                      (OPENPI_CONFIG_YAML, EXP_NAME, ノード数, resume/overwrite, preflight)
.env.local            個人設定・認証
                      (WANDB_API_KEY, CACHE_ROOT, CHECKPOINT_DIR, HF_LEROBOT_HOME)
configs/experiments/  実験の意味論
*.yaml                (データセット / モデル / バッチ / LR / サンプラ / GPU 数 /
                       base checkpoint)
```

YAML 側で実験の再現性を担保し、`.env*` は環境依存と個人差を吸収します。

### 5.2 YAML セクション

| セクション | 主なフィールド |
| --- | --- |
| `experiment` | `name`, `project_name`, `seed`, `wandb_enabled` |
| `dataset` | `repo_id`, `data_dir`, `assets_dir`, `asset_id`, `action_mode`, `prompt_from_task`, `fast_lerobot`, `lerobot_backend`, `video_backend`, `adapt_to_pi`, `convert_gripper` 等 |
| `model` | `type` (`pi0` / `pi05`), `paligemma_variant`, `action_expert_variant`, `action_dim`, `action_horizon`, `finetune_recipe` (`vision_lora` 等) |
| `training` | `batch_size`, `num_train_steps`, `pin_memory`, `prefetch_factor`, `eval_interval`, `save_interval`, `keep_period`, `lr_schedule` (`cosine_decay` など) |
| `gpu` | `num_gpus`, `base_gpus`, `fsdp_devices` |
| `scaling` | `base_batch_size`, `scale_batch_size`, `scale_learning_rate`, `scale_train_steps`, `workers_per_gpu` |
| `task_sampler` | `kind` (`uniform` 等), `alpha`, `ema_decay`, `min_prob` |
| `weight_loader` | `type` (`checkpoint` / `paligemma` / `noop`), `params_path` |
| `checkpoints` | `base_model.<type>` で base checkpoint URL を指定 |

スキーマの完全な仕様は `src/openpi/training/experiment_config.py` を参照してください
(`ExperimentConfig.from_yaml`)。

### 5.3 `_base_` による継承

YAML の先頭に `_base_: <ファイル名>` を書くと、同ディレクトリの YAML を
マージベースにできます (`load_yaml_with_inheritance` が再帰的に解決)。

```yaml
# configs/experiments/example_lora.yaml (抜粋イメージ)
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

### 5.4 参考 YAML (commit 対象テンプレ)

- `configs/experiments/example_base.yaml` — π0.5 fine-tune の最小構成
- `configs/experiments/example_lora.yaml` — LoRA 学習の構成例
- `configs/experiments/example_multinode.yaml` — multinode 用の GPU 数・スケーリング設定例

> これ以外の `*.yaml` は `.gitignore` でローカル専用扱いです。チームで共有する
> 場合は `example_*.yaml` を複製して新しい名前で運用するか、`.gitignore` の
> 除外ルールを追加してください。

### 5.5 tyro (`CONFIG_NAME`) 経路 (legacy)

`scripts/train.py` は `--config-yaml` 未指定時、最初の positional 引数を
tyro の TrainConfig 登録名とみなします。登録は `src/openpi/training/config.py`
にあります。一部例:

- `pi05_hsr`, `pi0_hsr`
- `pi0_aloha`, `pi05_aloha`, `pi0_libero` など
- 多数の fine-tune 派生

```bash
# Docker コンテナ内 (ローカル) の場合:
uv run scripts/train.py pi05_hsr --exp-name my_exp --overwrite
# HPC SLURM 単ノード launcher は tyro 経路です (§8.1 参照):
sbatch RUN-UV-SBATCH.sh
```

> tyro 経路は単一ノード開発用途に便利ですが、**マルチノード学習・Blackwell
> overlay・policy server はすべて YAML 前提** で組まれています。
> 新規の実験は YAML で設計してください。

---

## 6. ローカル環境

ローカル GPU マシンでの学習は **すべて Docker 経由** で行います。コンテナの
3 モードがあります (`train` / `train-gb200` / `deploy`)。ホスト側の事前準備は
§4 (`.env` / `.env.local` コピー) と Docker / NVIDIA Container Toolkit の
インストールのみです。

> **ホスト直実行 (pixi + uv で `scripts/train.py` を叩く) は一旦サポート外です。**
> ソース閲覧や型チェック用途であれば `pixi run sync` で venv を作っておくと
> IDE / LSP が動きやすくなりますが、学習は Docker を使ってください。

### 6.1 Docker: 非 Blackwell 学習 (`train`)

`docker/Dockerfile.train` をベースに CUDA 12.2 / Ubuntu 22.04 のイメージを作ります。
**Docker 経路では pixi は使わず**、コンテナ内で bare `uv` を使います
(ffmpeg 7 はコンテナビルド時に内蔵済み)。

**前提**

- Docker + NVIDIA Container Toolkit
  (未インストールなら `scripts/docker/install_docker_ubuntu22.sh` と
   `scripts/docker/install_nvidia_container_toolkit.sh` が参考になります)
- 環境変数:
  - `DEEP_PROJECT_NAME` (必須): compose project 名。コンテナ名は
    `${DEEP_PROJECT_NAME}_deep_1` となります
  - `DEEP_DATASET_PATH` (推奨): データセットのホスト側パス。コンテナ内 `/home/datasets` にマウント
  - `DEEP_CACHE_ROOT_HOST` (任意): uv / HF / tmp のホスト側キャッシュディレクトリ。未指定なら `./.docker_cache` に自動作成

**ビルド**

```bash
export DEEP_PROJECT_NAME=review
export DEEP_DATASET_PATH=/path/to/lerobot_datasets
./BUILD-DOCKER-CONTAINER.sh train
```

**起動**

```bash
./RUN-DOCKER-CONTAINER.sh train
# docker compose up -d の後、コンテナに exec で入る
```

起動するとコンテナ内で `docker/scripts/initialize-docker-container.sh` が走り、
以下が自動で行われます:

1. `/home/cache/*` (uv, HF, tmp, XDG) の作成
2. `.bashrc` 整備 (`${UV_PROJECT_ENVIRONMENT}/bin` と GB200 overlay env のソース。
   コンテナ内では既定 `/home/cache/venv` で、ホスト側 `.docker_cache/venv` に
   マッピングされます。ホストの `.venv/` とは意図的に分離されています)
3. 初回の `uv sync` (ロック済み環境をイメージ外で展開)
4. GB200 overlay 適用 (モードが `train-gb200` のときのみ)
5. 最後に `tail -f /dev/null` で常駐

exec 後はリポジトリルート (`/home/openpi`) にいるので、そのまま

```bash
uv run scripts/train.py --config-yaml configs/experiments/example_base.yaml --exp-name my_exp --overwrite
```

で学習できます。norm stats 計算も同様に `uv run ...` で。

**ボリュームマウント (`docker/docker-compose.yml`)**

| ホスト側 | コンテナ側 | 用途 |
| --- | --- | --- |
| リポジトリルート | `/home/openpi` | ソース (編集可) |
| `./docker/scripts` | `/home/docker_scripts` | init / overlay スクリプト |
| `${DEEP_CACHE_ROOT_HOST}` | `/home/cache` | uv / HF / tmp |
| `${DEEP_DATASET_PATH}` | `/home/datasets` | LeRobot データセット |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | X11 (GUI デバッグ用) |
| `/etc/passwd`, `/etc/group` (ro) | 同左 | ユーザ解決 |
| `/dev/` | `/dev/` | GPU / USB |

コンテナ内で使う主な CLI フラグ (`uv run scripts/train.py`):

| フラグ | 意味 |
| --- | --- |
| `--config-yaml PATH` | YAML パス (推奨, tyro と排他) |
| `--exp-name NAME` | run 名 (YAML `experiment.name` を上書き) |
| `--resume` | 既存チェックポイントから再開 |
| `--overwrite` | チェックポイントディレクトリを上書き作成 |
| `--checkpoint-base-dir PATH` | チェックポイントのベースを上書き (既定は `CHECKPOINT_DIR`) |
| `--assets-base-dir PATH` | assets ベースを上書き |
| `--seed N` | 乱数 seed の上書き |

norm stats の計算はコンテナ内で次のように:

```bash
JAX_PLATFORMS=cpu uv run python scripts/aggregate_stats_fast.py \
  --episodes-stats "${DATA_DIR}/meta/episodes_stats.jsonl" \
  --output-file "assets/<exp_name>/<repo_id>/norm_stats.json" \
  --chunk-dir "${DATA_DIR}/data/" \
  --action-column "action.relative" \
  --action-mode "relative"
```

`--action-mode` は YAML の `dataset.action_mode` と揃えてください
(現行 `relative` が推奨値)。

### 6.2 Docker: Blackwell (GB200) 学習 (`train-gb200`)

GB200 (cu128 + aarch64) では Dockerfile.train.gb200 を使います。コンテナ内で
`docker/scripts/apply-gb200-overlay.sh` が cu128 系 torch を自動的に overlay します。

**概要**

- ベースイメージ自体は `nvidia/cuda:12.2.2-devel-ubuntu22.04`
- 初回 `uv sync` で cu122 の torch (2.7.1) と jax (0.5.3) が入る
- その直後 `docker/scripts/apply-gb200-overlay.sh` が起動し、
  - `torch==2.10.0+cu128` / `torchvision==0.25.0+cu128` / `torchcodec==0.10.0+cu128`
    を PyTorch 公式 cu128 index から入れ直し
  - `${UV_PROJECT_ENVIRONMENT}/.gb200_overlay_env.sh` を作成 (cu128 torch lib と
    nvidia 各 lib を `LD_LIBRARY_PATH` に追加)
  - 同ファイルを `.bashrc` 経由で **インタラクティブシェル起動時** に読み込む
  - overlay を再適用するかどうかは
    `${UV_PROJECT_ENVIRONMENT}/.gb200_overlay_stamp` の内容で判定

**使い方**

```bash
export DEEP_PROJECT_NAME=gb200test
export DEEP_DATASET_PATH=/path/to/datasets

./BUILD-DOCKER-CONTAINER.sh train-gb200
./RUN-DOCKER-CONTAINER.sh train-gb200
# exec 後、cu128 が効いているか確認
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
# → 2.10.0+cu128 12.8 程度が出れば OK
```

**注意点**

- GB200 overlay で入る `LD_LIBRARY_PATH` は `.bashrc` 経由で読まれるため、
  `docker exec ... bash -lc '...'` のようにログインシェルを経由して
  コマンドを叩くのが安全です。もしくは
  `source ${UV_PROJECT_ENVIRONMENT}/.gb200_overlay_env.sh` をコマンド先頭で実行してください
  (コンテナ内ではデフォルトで `/home/cache/venv`)

### 6.3 Docker: HSR デプロイ (`deploy`)

Dockerfile.deploy は Ubuntu 20.04 / ROS Noetic ベースで、HSR ロボット連携
(catkin ワークスペース、hsr_data_msgs など) を内包します。

**ビルドに必要な追加情報**

```bash
export DEEP_PROJECT_NAME=hsrdeploy
export HSR_APT_USER=<apt user>      # HSR メーカー提供 apt レポ認証
export HSR_APT_PASSWORD=<apt pass>
./BUILD-DOCKER-CONTAINER.sh deploy
```

**起動**

```bash
# ロボット解決: ROBOT_NAME か HSR_IP のどちらかを指定
export ROBOT_NAME=hsrb107           # もしくは:
# export HSR_IP=192.168.1.2

# ROS_IP は起動時に interactive 選択 (複数 IF がある場合)
./RUN-DOCKER-CONTAINER.sh deploy
```

`RUN-DOCKER-COMMON.sh::openpi_resolve_deploy_network` の挙動:

- `HSR_IP` が未設定なら `ROBOT_NAME` を `getent hosts` → `avahi-resolve` の順で解決
- `ROS_IP` が未設定なら `ifconfig` / `ip addr` の結果から対話選択
- `ROS_MASTER_URI` は `http://${HSR_IP}:11311` に自動設定

**備考**

- deploy モードではコンテナ起動時に `initialize-ros-env.sh` が追加で走り、
  catkin build 等の初期化が行われます

---

## 7. モデルのデプロイ

HSR 実機への推論デプロイは、§6.3 でビルド・起動した **deploy Docker コンテナ**
の中で `roslaunch` を使って推論ノードを立ち上げる構成です。
別途 `server/Dockerfile` を使う policy server は、外部評価ツールなどから
WebSocket で推論リクエストを受ける用途向けに用意されています（§7.2）。

### 7.1 HSR 実機デプロイ (ROS / `roslaunch`)

前提:

- §6.3 の手順で deploy 用 Docker イメージをビルドし、コンテナを起動してその
  シェルに入っていること
- 学習済みチェックポイント (step ディレクトリ) がコンテナ内から参照できる場所
  (`/home/openpi/checkpoints/...` など) に置かれていること
- HSR 実機に有線/無線でネットワーク接続済みで、`ROS_MASTER_URI` が
  `http://${HSR_IP}:11311` に解決できていること (§6.3 のスクリプトで自動設定)

#### 7.1.1 ROS 環境の読み込み

コンテナ内で、catkin ワークスペースの setup を読み込みます:

```bash
source /root/catkin_ws/devel/setup.bash
```

コンテナ起動時に `initialize-docker-container.sh` が `.bashrc` へ追記している
ので、通常はログインシェル (`bash -l` / `exec bash`) に入り直せば自動で
読み込まれます。

#### 7.1.2 Python (uv) 環境の同期

コンテナ内で、OpenPI 側の Python 3.11 venv を作成/更新します:

```bash
cd /home/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
```

コンテナ初回起動時にも `warm_uv_environment` が自動同期を走らせますが、
手動で明示的に同期しておくと依存関係の差分を早期に検出できます。

#### 7.1.3 ROS パッケージのビルド

catkin ワークスペースをビルドします（初回と、`deploy/` 配下を更新した時）:

```bash
cd /root/catkin_ws
catkin build
source devel/setup.bash
```

#### 7.1.4 推論の起動

起動前に、以下のファイルが目的のモデル／コントローラ設定になっていることを
確認します:

- `src/openpi/training/config.py` — tyro 経路で使う場合のみ。使う `TrainConfig`
  の登録内容をチェック
- `deploy/hsr_data_collection/hsr_data_collection/config/hsr_data_collection_config.yaml`
  — DualShock 3 / 4 などコントローラ種別の設定

続いて DualShock 3 または 4 コントローラを HSR 本体へ接続します。

**A. tyro の config 名を指定して起動**

```bash
roslaunch hsr_openpi hsr_openpi.launch \
  config_name:=pi05_hsr \
  checkpoint_dir:=/home/openpi/checkpoints/pi05_hsr/my_experiment/100000
```

**B. YAML をそのまま指定して起動 (推奨)**

実験 YAML を与えると、学習時のモデル/正規化設定をそのまま推論に使えます:

```bash
roslaunch hsr_openpi hsr_openpi.launch \
  config_yaml:=configs/experiments/your_run.yaml \
  checkpoint_dir:=/path/to/your_run/100000
```

主な起動引数（`deploy/hsr_openpi_deploy/launch/hsr_openpi.launch`）:

| 引数 | 既定値 | 役割 |
| --- | --- | --- |
| `config_name` | `pi0_hsr_low_mem_finetune` | tyro の TrainConfig 登録名 |
| `config_yaml` | `""` | 実験 YAML。指定時は `config_name` より優先 |
| `checkpoint_dir` | (launch で指定) | 推論に使う step ディレクトリ |
| `update_freq` | `10` | 推論ループの Hz |
| `adopted_action_chunks` | `10` | 推論結果から採用する action chunk 数 |
| `upsample`, `upsample_hz`, `upsample_method` | `true`, `100`, `spline` | action の時間方向アップサンプル設定 |
| `action_smoothing`, `ema_alpha`, `ma_window` | `ema`, `0.2`, `5` | action 平滑化 (`ema` / `moving_average` 等) |
| `smooth_gripper`, `smooth_base` | `false`, `false` | gripper / 台車を平滑化するか |
| `gripper_mode` | `hybrid` | gripper 制御モード |
| `save_exec_trace` | `false` | 実行トレースを保存するか |

#### 7.1.5 実行時の言語命令の更新

推論中に言語命令を差し替える場合は ROS サービスを叩きます:

```bash
rosservice call /hsr_openpi/update_instruction "message: 'Open the oven toaster'"
```

#### 7.1.6 アクション実行の開始

コントローラの **左十字キー (Dpad Left)** を押すとアクション実行が開始されます。

### 7.2 policy server (`server/Dockerfile`)

トップレベルの `server/` には、WebSocket ベースの推論サーバを別プロセス/別
コンテナとして立てるためのビルド資産があります。評価ツール・外部クライアント
からのリモート推論呼び出しなどに利用できます。

詳細 (イメージビルド手順・`server/entrypoint.sh` が読む環境変数・checkpoint
埋め込み YAML の自動検出・hot reload / health check エンドポイント) は
`server/Dockerfile`, `server/entrypoint.sh`, `server/serve_hsr_policy_ws.py`,
`src/openpi/serving/websocket_policy_server.py` を参照してください。

---

## 8. HPC — SLURM

### 8.1 単一ノード (`RUN-UV-SBATCH.sh` / `RUN-UV-SBATCH-8GPU.sh`)

どちらも **tyro `CONFIG_NAME` 経路** です。YAML モードでは使えません。

最小 `.env` / `.env.local`:

```bash
# .env (実験)
CONFIG_NAME=pi05_hsr        # src/openpi/training/config.py に登録がある名前
EXP_NAME=my_slurm_run

# .env.local (個人)
WANDB_API_KEY=...
DATA_DIR=/path/to/lerobot_datasets
HF_LEROBOT_HOME=/path/to/lerobot_cache
CHECKPOINT_DIR=/path/to/checkpoints
CACHE_ROOT=/path/to/cache_root
```

投入:

```bash
sbatch RUN-UV-SBATCH.sh        # 1 node 1 GPU / 24h
sbatch RUN-UV-SBATCH-8GPU.sh   # 1 node 8 GPU
```

スクリプトは内部で以下を実行します:

1. `module load cuda` (Environment Modules がある場合)
2. `RUN-UV-COMMON.sh` を読み込んで `.env` → `.env.local` を順にロード
3. `pixi run uv sync`
4. `aggregate_stats_simple.py` で norm_stats を計算
5. `pixi run uv run scripts/train.py "${CONFIG_NAME}" --exp-name=... --overwrite`

### 8.2 multi-node (`RUN-UV-SBATCH-OPENPI-MULTINODE.sh`)

YAML モード前提の multi-node スクリプトです。内部で自分自身を `sbatch` に
再投入し、`script_distribute_slurm_tasks.sh` を経由して各ランクで srun を投げます。

最小 `.env` / `.env.local`:

```bash
# .env (実験)
OPENPI_CONFIG_YAML=configs/experiments/example_multinode.yaml
EXP_NAME=pi05_hsr_task689_run1

OPENPI_NUM_NODES=4
OPENPI_GRES=gpu:8
OPENPI_CPUS_PER_TASK=240
OPENPI_SBATCH_ARGS="--partition=<your-partition> --mem=1490000M --time=336:00:00"
OPENPI_RESUME=0
OPENPI_OVERWRITE=1

OPENPI_PREFLIGHT_ENABLE=1

# .env.local (個人)
WANDB_API_KEY=...
CACHE_ROOT=/groups/.../tmp_storage
CHECKPOINT_DIR=/groups/.../checkpoints
HF_LEROBOT_HOME=/groups/.../lerobot_cache
```

投入:

```bash
bash RUN-UV-SBATCH-OPENPI-MULTINODE.sh
```

YAML の `gpu.num_gpus` は `OPENPI_NUM_NODES * (OPENPI_GRES の GPU 数)` と一致
させるのが基本。ずれているとランチャ側で警告が出ます。

### 8.3 Slurm 対話ノード取得 (`run_interactive_node.sh`)

開発・デバッグで Slurm 対話ノードを確保するラッパです。`salloc` 相当。

```bash
./run_interactive_node.sh                      # 既定: 8 GPU / 192h / 1900G mem
./run_interactive_node.sh -g 1 -t 1:00:00      # 1 GPU / 1 時間
./run_interactive_node.sh -p <your-partition> -g 8 -t 4:00:00 -c 56
./run_interactive_node.sh --cpu-only           # GPU なし (connector ノード用)
```

主なフラグ:

| フラグ | 説明 |
| --- | --- |
| `-p, --partition PART` | Slurm partition (既定 `<your-partition>`) |
| `-g, --gpus N` | GPU 数 (既定 8) |
| `-c, --cpus N` | CPUs per task (既定 224) |
| `-t, --time HH:MM:SS` | Wall time (既定 `192:00:00`) |
| `-m, --mem SIZE` | Memory (既定 `1900G`) |
| `--cpu-only` | GPU を要求せず connector ノードに乗る |

取得したシェルで学習を走らせる場合は、`sbatch RUN-UV-SBATCH.sh` などの HPC
launcher (§8.1 / §8.2) をそのまま投げるのが基本です。
launcher 内部で `pixi run uv sync` / `pixi run uv run ...` が自動的に呼ばれます。

---

## 9. HPC — PBS (qsub)

ABCI を想定した PBS + MPI (OpenMPI / HPCX) 経路です。

### 9.1 推奨: `RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh`

YAML モード専用。HPCX を `module load` し、自身の PBS ジョブ内で直接
`mpirun` を叩くシンプル構成です。

最小 `.env` / `.env.local`:

```bash
# .env (実験)
OPENPI_CONFIG_YAML=configs/experiments/example_multinode.yaml
EXP_NAME=pi05_hsr_task6_4nodes_run1
OPENPI_RESUME=1            # 既存から再開する場合
OPENPI_OVERWRITE=0

# .env.local (個人)
WANDB_API_KEY=...
DATA_DIR=/groups/.../lerobot_datasets
HF_LEROBOT_HOME=/groups/...
CHECKPOINT_DIR=/groups/.../${USER}/AiroPi/checkpoints
CACHE_ROOT=/groups/.../tmp_storage
```

PBS ヘッダ (スクリプト先頭) で select/walltime 等が指定されています。ノード数は
`select=N:ncpus=192:mpiprocs=8` から自動計算され、1 ノード 8 GPU 固定の
前提です (YAML の `gpu.num_gpus` は `N*8` と揃える)。

投入:

```bash
qsub RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh
# もしくは PBS ヘッダを上書きしたければ:
qsub -l select=4:ncpus=192:mpiprocs=8 RUN-UV-QSUB-OPENPI-MULTINODE-SIMPLE.sh
```

起動後、以下のログが各ランクから出れば成功:

```
[INFO] MASTER_ADDR=node-abc
[INFO] MASTER_PORT=12345
[INFO] NUM_NODES=4
[INFO] CONFIG=configs/experiments/example_multinode.yaml
```

前処理で base model の事前ダウンロードと norm_stats 再計算
(`scripts/aggregate_stats_fast.py`) が mpirun 起動前に単一プロセスで走ります
(`pixi run uv run python` 経由)。

### 9.2 高機能版: `RUN-UV-QSUB-OPENPI-MULTINODE.sh`

UCX チューニングや preflight parquet チェック、`OPENPI_MPI_TASKS_PER_NODE` の
ノードあたりタスク数自由化などに対応しています。`.env` で以下の変数を追加できます:

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

投入:

```bash
bash RUN-UV-QSUB-OPENPI-MULTINODE.sh
```

### 9.3 Singularity

Docker が使えない HPC 向けの代替として、`docker/openpi.def` をベースに
`airopi.sif` をビルドする経路が用意されています。

```bash
./BUILD_SINGULARITY.sh                        # airopi.sif を生成
qsub RUN-SINGULARITY.sh                       # 学習 + uv sync
qsub RUN-SINGULARITY-FT.sh                    # fine-tune 向け
qsub RUN-SINGULARITY-COMPUTE-SIMPLE-NORM.sh   # norm_stats のみ
```

- 各スクリプトは `SIF_PATH` と `WORK_DIR` を引数/環境変数で上書きできます
- `--bind /groups/<your-project>/dataset:/groups/<your-project>/dataset` が ABCI 固有で
  ハードコードされているため、他サイトでは該当行を書き換えてください
- env ファイルの読み方は Family B (§4.2 参照) で、`.env.local` があるとそれだけ
  が使われる点に注意

---

## 10. ワークフロー横断 Tips

### 10.1 norm stats スクリプトの使い分け

| スクリプト | 特徴 | 用途 |
| --- | --- | --- |
| `scripts/aggregate_stats_fast.py` | JAX で並列集計、大規模データセット向け | HPC / multinode 推奨 |
| `scripts/aggregate_stats_simple.py` | pure Python、小規模・デバッグ向け | 単一ノード / 小規模 |
| `scripts/compute_norm_stats.py` | tyro CONFIG_NAME を引数に取るシンプル経路 | tyro モードのみ |

### 10.2 checkpoint ディレクトリ構成

学習 1 step ごとの保存は Orbax で以下を出します:

```
${CHECKPOINT_DIR}/<exp_name>/
 └── <step>/
     ├── train_state/             # オプティマイザ状態等
     ├── params/                  # 推論用パラメータ
     ├── assets/<asset_id>/norm_stats.json
     └── experiment_config/experiment_config.yaml   # YAML モード時のみ
```

`experiment_config.yaml` には `_base_` を解決したマージ済み YAML が
埋め込まれるので、deploy 時は step ディレクトリを丸ごと渡せば復元可能です
(§7.2 の policy server が自動検出)。

### 10.3 wandb 連携

- `.env.local` に `WANDB_API_KEY` を入れれば ON
- YAML `experiment.wandb_enabled: false` で OFF
- マルチノード時は rank0 のみが本物の run、他 rank は `disabled` モードで
  init されます

### 10.4 distribute 環境変数の解決順序

`src/openpi/training/distributed.py` は以下の env を順に探します。MPI ランタイム
(`OMPI_COMM_WORLD_RANK` など) からも自動で復元します。

| 標準名 | MPI フォールバック |
| --- | --- |
| `RANK` | `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `PMIX_RANK`, `MV2_COMM_WORLD_RANK` |
| `LOCAL_RANK` | `OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, ... |
| `WORLD_SIZE` | `OMPI_COMM_WORLD_SIZE`, ... |
| `LOCAL_WORLD_SIZE` | `OMPI_COMM_WORLD_LOCAL_SIZE`, ... |
| `MASTER_ADDR`, `MASTER_PORT` | 明示必須 |

### 10.5 `pixi run uv run` vs bare `uv run` の整理

ローカル学習は Docker 経由の bare `uv run ...` に固定されているため、基本的に
ユーザが `pixi run` を直接叩く場面はありません。

- **Docker コンテナ内 / Singularity 内**: bare `uv run ...` で OK (ffmpeg は
  イメージに内蔵)
- **HPC launcher の内部処理**: `RUN-UV-COMMON.sh` が `pixi run uv sync` /
  `pixi run uv run ...` を自動で叩く。ログインノードに pixi+uv が
  インストールされている前提
- **例外**: `RUN-UV-QSUB-OPENPI-MULTINODE.sh` 高機能版は歴史的経緯で bare
  `uv run` を残しています (統一候補)

---

## 11. `.env` / `.env.local` 変数早見表

### 11.1 `.env` (実験・ランチャ設定, `.env.example` に対応)

| 変数 | スコープ | 説明 |
| --- | --- | --- |
| `OPENPI_CONFIG_YAML` | YAML 経路 | 学習 YAML のパス (相対/絶対) |
| `EXP_NAME` | 全経路 | run 識別名 |
| `CONFIG_NAME` | tyro 経路 | tyro の登録名 (legacy) |
| `DATASET_NAME` | tyro 経路 | データセットサブパス (legacy) |
| `OPENPI_RESUME` | 学習 | `1` なら `--resume` (overwrite と両立不可) |
| `OPENPI_OVERWRITE` | 学習 | `1` なら `--overwrite` |
| `OPENPI_NUM_NODES` | SLURM/PBS | ノード数 |
| `OPENPI_GRES` | SLURM | `gpu:8` など |
| `OPENPI_SBATCH_ARGS` | SLURM | 追加 SBATCH オプション |
| `OPENPI_CPUS_PER_TASK` | SLURM | タスクあたり CPU 数 (既定 240) |
| `OPENPI_MPI_TASKS_PER_NODE` | PBS 高機能版 | ノードあたり MPI ランク数 |
| `OPENPI_MPI_MAP_BY` | PBS 高機能版 | OpenMPI の map 指定 |
| `OPENPI_MPI_LAUNCHER` | PBS 高機能版 | `mpirun` / `mpiexec` |
| `OPENPI_MPI_MODULE` | PBS 高機能版 | 読み込む module (既定 `hpcx/2.20`)。SIMPLE 版は `hpcx/2.20` をハードコード |
| `OPENPI_MPI_SHARED_TMPDIR` | PBS 高機能版 | mpi 共有 tmp |
| `OPENPI_MPI_ARGS` | PBS 高機能版 | 追加 mpirun オプション |
| `OPENPI_QSUB_ARGS` | PBS 高機能版 | 追加 qsub オプション |
| `OPENPI_QSUB_SELECT` | PBS 高機能版 | qsub `-l select=...` |
| `OPENPI_MPI_ASSIGN_UCX_NET_DEVICES` | PBS 高機能版 | `1` ならランク毎に `UCX_NET_DEVICES` 設定 |
| `OPENPI_MPI_NDR_COUNT` | PBS 高機能版 | HCA 数 (既定 8) |
| `OPENPI_AGGREGATE_STATS_ENABLE` | PBS 高機能版 | `1` で norm_stats 再計算を事前実行 |
| `OPENPI_PREFLIGHT_ENABLE` | SLURM/PBS 高機能版 | `1` で parquet ノード間一貫性チェック |
| `OPENPI_PREFLIGHT_SAMPLES_PER_NODE` | SLURM/PBS 高機能版 | preflight サンプル数 |

### 11.2 `.env.local` (個人・マシン設定, `.env.local.example` に対応)

| 変数 | スコープ | 説明 |
| --- | --- | --- |
| `WANDB_API_KEY` | 全経路 | wandb ロギング (未設定ならログ送信されない) |
| `WORKING_DIR` | uv 系 | 作業ディレクトリ (既定: スクリプト配置先) |
| `CACHE_ROOT` | 学習全般 | uv/HF/tmp/XDG などのベース (計算ノードから書ける先) |
| `CHECKPOINT_DIR` | 学習全般 | チェックポイント保存先ルート |
| `HF_LEROBOT_HOME` | 学習全般 | LeRobot の HF cache |
| `DATA_DIR` | 学習全般 | LeRobot データセットのルート (通常は YAML から推論、必要時のみ上書き) |

### 11.3 Docker 系

| 変数 | スコープ | 説明 |
| --- | --- | --- |
| `DEEP_PROJECT_NAME` | Docker | compose project 名 (必須) |
| `DEEP_DATASET_PATH` | Docker | データセットのホスト側パス |
| `DEEP_CACHE_ROOT_HOST` | Docker | ホスト側キャッシュ (既定: `./.docker_cache`) |
| `DEEP_IMAGE_TAG` | Docker | イメージタグ (既定は mode 名) |
| `DEEP_DOCKERFILE` | Docker | 使用 Dockerfile (mode が自動設定) |
| `OPENPI_PROJECT_PYTHON` | Docker | 既定 `3.11` |
| `OPENPI_ENABLE_ROS_INIT` | Docker | `auto` (既定) / `1` / `0` |
| `OPENPI_PYTHON_OVERLAY_HOOK` | Docker (GB200) | overlay スクリプトパス (Dockerfile で自動設定) |
| `HSR_APT_USER`, `HSR_APT_PASSWORD` | Docker deploy | HSR apt 認証 (必須) |
| `ROBOT_NAME`, `HSR_IP`, `ROS_IP`, `ROS_MASTER_URI` | Docker deploy | ROS 接続 |

### 11.4 policy server (`server/`)

| 変数 | スコープ | 説明 |
| --- | --- | --- |
| `POLICY_CHECKPOINT_DIR` | server | 推論する step ディレクトリ (必須) |
| `POLICY_CONFIG_NAME`, `POLICY_CONFIG_YAML` | server | どちらか必須 (checkpoint に `experiment_config.yaml` が埋め込まれていれば省略可) |
| `POLICY_SERVER_HOST`, `POLICY_SERVER_PORT` | server | 既定 `0.0.0.0:8000` |
| `POLICY_DEFAULT_PROMPT` | server | prompt フォールバック |
| `POLICY_RECORD_DIR` | server | 記録保存先 |
| `POLICY_PYTORCH_DEVICE` | server | `cuda` / `cpu` |

### 11.5 env ファイル解決の上書き

| 変数 | スコープ | 説明 |
| --- | --- | --- |
| `HSR_OPENPI_ENV_FILE` | Family A / Family B の多く | `.env` の代わりに読むファイルパス |
| `AIROPI_ENV_FILE` | `RUN-SINGULARITY-FT.sh` のみ | 上記と同役割の別名 (要修正候補) |

---

## 12. トラブルシューティング

### 12.1 Docker コンテナ内で `uv sync` が失敗する

- ネットワークエラーの場合は再実行で通ることが多い
- HF / LeRobot の LFS に触りたくない場合は `GIT_LFS_SKIP_SMUDGE=1 uv sync`
- ホスト側の `.venv/` が root 所有で残っているとボリュームマウントと干渉する
  ことがあります。Docker 側の venv は `${UV_PROJECT_ENVIRONMENT}` (既定
  `/home/cache/venv` = ホスト `.docker_cache/venv`) に隔離されているので、
  ホストの `.venv/` は削除して問題ありません

### 12.2 GB200 overlay が再適用されない・cu128 にならない

- `${UV_PROJECT_ENVIRONMENT}/.gb200_overlay_stamp` の内容で現行 overlay を
  検出しています。環境変数 (`OPENPI_GB200_TORCH_VERSION` など) を変えると
  次回コンテナ起動時に自動で再 install されます。強制的に再実行したい場合は
  stamp ファイルを削除してコンテナを再起動してください

### 12.3 deploy コンテナで ROS 解決に失敗

- `ROBOT_NAME` が `.local` 名のとき `avahi-resolve -4 --name ...` が効く必要があります
  (`apt install avahi-utils`)
- 明示的に IP を入れたいときは `.env` / シェルで `HSR_IP=...` を指定


---

## 13. 付録


### 13.1 参考 YAML 一覧 (tracked)

`configs/experiments/` 配下で **commit 対象** は以下 3 本のみです:

- `example_base.yaml` — π0.5 fine-tune の最小構成
- `example_lora.yaml` — LoRA (vision / action) 構成例
- `example_multinode.yaml` — multinode 用 (`gpu.num_gpus` を大きめに設定した例)

個別の実験 YAML を共有したい場合は、`example_*.yaml` からコピーして新しい名前を付け、
`.gitignore` の除外ルールを追加するか、`example_` プレフィックスに揃えてください。

### 13.2 Python / 主要依存の版

- Python 3.11.x
- jax 0.5.3 (cu12)
- torch 2.7.1 (非 Blackwell) / 2.10.0+cu128 (Blackwell overlay)
- flax 0.10.2, orbax-checkpoint 0.11.13
- transformers 4.53.2
- lerobot (git rev 固定, pyproject.toml 参照)
- pixi: ffmpeg ≥ 7 < 8, pkg-config, compilers, cython (conda-forge)

### 13.3 公開予定のモデル重み

HSR 向け π0.5 の以下のチェックポイントを公開予定です。各エントリは
`scripts/train.py` が出力する `${CHECKPOINT_DIR}/<exp_name>/<step>/` の step
ディレクトリ (構造は §10.2 を参照) に対応します。デプロイ launcher (§7.1) の
`checkpoint_dir:=...` 引数や policy server (§7.2) の `POLICY_CHECKPOINT_DIR`
にこの step ディレクトリを指定して読み込みます。

| 学習段階 | run 名 (step) | タスク |
| --- | --- | --- |
| 事前学習 | `pi05_hsr_75tasks_fast_multinode_8nodes_vision_lora_action_full` (step 250,000) | HSR 75 タスク。π0.5 を vision LoRA + action expert full で 8 ノード学習 |
| 事前学習 + 事後学習 | `pi05_hsr_exercise_ph1_0405_lora_pretrain` (step 49,999) | テーブル上の pick & place (1 タスク) |
| 事前学習 + 事後学習 | `pi05_hsr_task6891011_level12_260304_finetune_68tasks_full_pi05` (step 200,000) | ボトル移動、ボトル 2 本移動、箱移動、電子レンジ開閉、カップ移動 (5 タスク) |

公開された重みは R2 バケット `airoa-oss-hsr-moma-5k` に格納されています。以下の
各パスは step ディレクトリ全体
([チェックポイントのディレクトリ構成](#102-チェックポイントのディレクトリ構成)) です。
ディレクトリごとダウンロードし、`checkpoint_dir:=...`
([ロボット実機へのデプロイ](#71-ロボット実機へのデプロイ-ros--roslaunch)) または
`POLICY_CHECKPOINT_DIR` ([ポリシーサーバ](#72-ポリシーサーバ-serverdockerfile)) に
指定してください。

| run 名 (step) | ダウンロードパス |
| --- | --- |
| `pi05_hsr_75tasks_fast_multinode_8nodes_vision_lora_action_full` (step 250,000) | `s3://airoa-oss-hsr-moma-5k/checkpoints/pretrained/pi05_hsr_75tasks_fast_multinode_8nodes_vision_lora_action_full/250000/` |
| `pi05_hsr_exercise_ph1_0405_lora_pretrain` (step 49,999) | `s3://airoa-oss-hsr-moma-5k/checkpoints/finetuned/pi05_hsr_exercise_ph1_0405_lora_pretrain/49999/` |
| `pi05_hsr_task6891011_level12_260304_finetune_68tasks_full_pi05` (step 200,000) | `s3://airoa-oss-hsr-moma-5k/checkpoints/finetuned/pi05_hsr_task6891011_level12_260304_finetune_68tasks_full_pi05/200000/` |
