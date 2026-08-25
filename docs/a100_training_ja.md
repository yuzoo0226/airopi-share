# A100 クラスタで学習する

GB10（DGX Spark）で 6.9 s/step だった pi0.5 のファインチューニングを、A100 の
Slurm クラスタへ移した記録と手順。**2.3 s/step になり、30k ステップが 58 時間から
19 時間になった。**

対象クラスタ: `aquamarine`（A100-SXM4-80GB、Ubuntu 22.04、x86_64、driver 570）。
パーティションは `part_20gb` / `part_40gb` / `part_80gb` で、いずれも時間制限なし。

---

## 1. なぜ移すと楽になったか

GB10 で苦労した点の **ほとんどが x86_64 では発生しない**。

| GB10 (aarch64, sm_121) | A100 (x86_64, sm_80) |
| --- | --- |
| jaxlib 0.5.3 が sm_121 で abort → 0.6.2 に上げる | `jax[cuda12]==0.5.3` が**そのまま動く** |
| orbax 0.11.13 が jax 0.6 と非互換 → 0.11.14 | ピン留め不要 |
| `chex` が jax を 0.7 に引き戻す → `PIP_CONSTRAINT` | 不要 |
| numpy<2 制約 | 不要 |
| `apply_sm121_jax_overlay.sh` | 不要 |

つまり `uv sync --frozen` だけで環境が揃う。**ただし後述の `av` を除く。**

---

## 2. セットアップ手順

### 2.1 SSH 鍵

パスワード認証しかない状態では自動化できないので、最初に鍵を入れる。

```bash
ssh-copy-id yano21@150.69.197.6
```

**通常のターミナルで実行すること。** Claude Code の `!` 実行やスクリプト経由では
パスワードプロンプトに応答できず、空入力として弾かれる。

### 2.2 uv と Python 3.11

openpi は `requires-python = ">=3.11.0, <3.12"` で、クラスタの system python は
3.10.6。root 権限は無いので uv でユーザ空間に入れる。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p ~/usr/airopi && cd ~/usr/airopi
git clone -b devel/ros2 https://github.com/yuzoo0226/airopi-share.git
cd airopi-share
UV_PROJECT_ENVIRONMENT=$HOME/usr/airopi/venv uv sync --frozen --no-install-package av
```

### 2.3 `av` — ffmpeg の沼

`uv sync --frozen` は **必ず失敗する**。

```
× Failed to build `av==14.4.0`
  It is EXPECTED that it will fail. You are REQUIRED to use ffmpeg 7.
  Package libavformat was not found in the pkg-config search path.
```

`av` 14.4.0 は PyPI に **ホイールが無く**、ソースビルドが ffmpeg 7 の開発ヘッダを
要求する。root が無いので apt では入れられない。

`--no-install-package av` でスキップしてもだめで、`lerobot` が import 時に
`av` を読む（`lerobot/common/datasets/video_utils.py:24`）。

**対処:** ホイールのあるバージョンを入れる。`lerobot` の要件は `av>=14.2.0` で、
16.1.0 以降にホイールがある。

```bash
cd $HOME   # プロジェクト内で実行すると override-dependencies が効いてしまう
uv pip install --python ~/usr/airopi/venv/bin/python "av==16.1.0"
```

このデータセットは動画を使わない（PNG を parquet に格納）ので、`av` は import
されるだけで実際には使われない。

### 2.4 データの転送

```bash
rsync -a --delete <local>/datasets/lerobot_datasets/hsr_gazebo_pick \
    yano21@150.69.197.6:~/usr/airopi/datasets/lerobot_datasets/
rsync -a <local>/checkpoints/airoa-pi05-hsr-base \
    yano21@150.69.197.6:~/usr/airopi/checkpoints/
scp <local>/assets/example_hsr_pick_gazebo/lerobot_datasets/hsr_gazebo_pick/norm_stats.json \
    yano21@150.69.197.6:~/usr/airopi/airopi-share/assets/example_hsr_pick_gazebo/lerobot_datasets/hsr_gazebo_pick/
```

**`--delete` を付けること。** データセットを作り直すとエピソード数が減ることが
あり（破損エピソードの破棄など）、付けないと古いインデックスのファイルが孤児と
して残る。

---

## 3. 実行

**必ず `sbatch` で投入する。** ログインノードで直接実行すると、スケジューラの外で
GPU を掴むことになる。

```bash
cd ~/usr/airopi/airopi-share
sbatch -p part_80gb scripts/a100/train_pick.sh
```

`scripts/a100/train_pick.sh` がやっていること:

* `#SBATCH --gres=gpu:a100:1` でフル A100 を1枚要求
* `HF_LEROBOT_HOME` を設定（**これが無いと Hugging Face に問い合わせに行って
  `RepositoryNotFoundError` で死ぬ**。ローカルの 7 GB は一切参照されない）
* `PYTHONUNBUFFERED=1`（後述）
* `num_workers` を 0 に上書き（後述）
* YAML の `data_dir` / `params_path` を sed でこのマシンのパスに書き換えた派生
  設定を生成し、書き換え後のパスが実在するか検査してから起動
* `--checkpoint-base-dir` を明示

### 3.1 監視

Slurm は **失敗したジョブも成功したジョブも `squeue` から消えるだけ**で区別が
つかない。

```bash
scripts/a100/watch_slurm_job.sh <job_id> 600
```

キューから消えた瞬間に `sacct` で終了状態を確認し、`FAILED` / `TIMEOUT` /
`OUT_OF_MEMORY` / `NODE_FAIL` を個別に名指しして `.out` の末尾を出す。
`RUNNING` のまま進捗が止まるケースも検出する。

### 3.2 評価

学習はクラスタ、シミュレータはワークステーションにしか無いので、評価のたびに
チェックポイント（9.3 GB）を持ち帰る。

```bash
scripts/a100/fetch_and_eval.sh 10000 20
```

---

## 4. ハマった点

### 4.1 `HF_LEROBOT_HOME` 未設定（ジョブ 7898、15 秒で FAILED）

```
huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error.
Repository Not Found for url: .../datasets/lerobot_datasets/hsr_gazebo_pick/refs
```

`FastLeRobotDatasetMetadata` は `HF_LEROBOT_HOME/<repo_id>` を見る。GB10 の
コンテナはイメージに設定してあったので、設定ファイルにもコマンドにも現れず、
移植時に落ちやすい。

### 4.2 dataloader ワーカーが起動できない（ジョブ 7899）

```
multiprocessing/synchronize.py __setstate__ -> SemLock._rebuild
FileNotFoundError: [Errno 2] No such file or directory
```

openpi は `num_workers > 0` のとき `spawn` を強制する
（`data_loader.py:577`）。spawn された子プロセスが渡されたセマフォの再構築に
失敗する。`/dev/shm` は 252 GB あり、ジョブの外ではセマフォも作れるので、
リソース制限ではなくジョブの名前空間の問題。

**これが一番たちが悪い。クラッシュしない。** openpi が失敗バッチを catch して
スキップするので、ジョブは `RUNNING` のまま、GPU 95%、`sacct` も正常、
`Skipping bad batch (error N/100)` を数秒ごとに出しながら**何も学習しない**。
`squeue` / `sacct` / `nvidia-smi` の全てが健全と表示する。

対処は `num_workers: 0`。A100 では 2.3 s/step 出ているので実害は無い。

### 4.3 loss が1行も出ない

`train.py` は loss を `pbar.write()` で書く。これは logging を経由せず Python の
stdout バッファに入り、Slurm はそれをファイルにブロックバッファリングする。
結果、tqdm の進捗行（こちらは logging 経由なので出る）だけが延々と流れ、
**loss は何時間走らせても1行も出ない。**

`PYTHONUNBUFFERED=1` で解決。

これは些細な設定に見えて、実際には **19 時間の学習が壊れていることに気づけるか
どうか** を左右した。§5 を参照。

---

## 5. 投入したら必ず step 0 の loss を読む

**pi05 HSR 学習の step 0 loss は 0.15〜0.22 が正常値。**

| データ | step 0 loss |
| --- | --- |
| 74 エピソード | 0.2147 |
| 247 エピソード | 0.1751 |
| 541 エピソード | **522160192** |
| 536 エピソード（破損除去後） | **0.1512** |

541 エピソードのデータセットには、物理的にありえない関節値を持つエピソードが
5 本混ざっていた。

```
wrist_flex_joint = -2.79 rad   (可動域 -1.92)   ← 3本、全フレーム破損
hand_motor_joint = -1.70, 2.37 (可動域 -0.798〜1.24)
```

339 フレーム、全体の 0.6%。Gazebo を落とす数値爆発（`ros2_worklog_ja.md` §6 の
`omni_base_controller` が 2.37e+48 を出すやつ）がアームに出たもの。分布から
13σ 外れた状態量を学習済みトランスフォーマーに入れると出力が発散する。

**どこにも警告が出ない。** bag の変換は無警告で通り、エピソードは正常に見え、
成功フィルタも通り（把持自体は成功している）、`param_norm` も正常値、学習は
フルスピードで進み GPU も 95%。**唯一の証拠が step 0 の loss 1行**で、しかも
§4.3 のバッファリングでそれすら出ない状態だった。

対処として `rosbag2_to_lerobot.py` に URDF 可動域チェックを入れた（余裕 0.2 rad、
違反エピソードは関節名と値を出して破棄）。

### 切り分けの目安

* `top_task_loss` を見る。破損なら **1〜2 個のタスクだけ**が桁違いに高い
  （実測: タスク5=6.4e8, 3=2.9e8 に対しタスク4=6e4）。正規化やチェックポイントの
  問題なら全タスクが一様に高い
* `param_norm=2228.3137` はベースモデルの読み込みが正常なことしか意味しない
  （`image_encoder_mode` の誤読なら 2019.61）

---

## 6. 訓練 loss のスパイクは指標にならない

学習が健全でも、訓練 loss は綺麗なステップとスパイクが交互に出る。

```
Step 700:    0.0177     Step 900:     0.0170
Step 750:   22367       Step 950:  266010
Step 800: 2073552       Step 1000:    0.0181
```

原因は **エピソード先頭のアーム過渡**。開始直後はアームが指令姿勢に達しておらず
`action = 指令 - 実測 = -π/2 - 0 = -1.5708` になる。`wrist_flex` の std が
0.023 なので |z| = 68 まで増幅される。各エピソードの先頭 13% 程度に必ず存在する。

**これは正当なデータで、除外してはいけない。** 推論時も同じ状況が起きる
（リセットでアームを 1.2 秒かけて配置し、その直後にポリシーへ制御が渡る）。

害が無いことは確認済み:

* `clip_gradient_norm: 1.0` が既定で有効（`optax.clip_by_global_norm`）。ログの
  `grad_norm=2407` は**クリップ前**の値で、実際の更新は常にノルム 1.0
* `param_norm` は安定（2228.3843 → 2228.5222）
* 検証 loss は明確に低下（0.2041 → 0.0135）

**健全性は `val/val_loss` で見ること。** 検証データにはスパイク源のバッチが
含まれないので、モデルの実力を正確に表す。1000 ステップごとに出る。

### 統計的な外れ値検出を使わないこと

今回、除外すべき破損（可動域外）と除外してはいけない過渡（|z|=68）が同時に
存在した。**可動域という物理的な基準で切ったから区別できた。** z スコアで
切っていれば、正当な過渡も一緒に捨てて、立ち上がりの弱いモデルになっていた。

---

## 7. 正規化の標準偏差に下限を入れる

`aggregate_stats_fast.py --min-std 0.01`（既定値）。

正規化は `(x - mean) / (std + 1e-6)`。このタスクは `arm_roll` と `wrist_roll` を
常に 0.0 で固定指令しているため、実測 std がコントローラのジッタそのもの
（0.0002 / 0.0001）になる。これで割ると 100 分の1ラジアンが |z| = 235 になる。

**データが少ないうちは発生しない。** 74 エピソードではこれらの関節がまったく
動かず std が厳密に 0 だったため、分子も 0 で無害だった。エピソードを増やして
わずかなジッタが入った瞬間に、0.0001 で割る形になる。

下限 0.01 で `arm_roll` は |z| 234.6 → 4.2、`wrist_roll` は 235.6 → 2.2 になる。
**厳密に 0 の次元は触らない**（padding か真に定数で、openpi が既に正しく扱う）。
本当に大きく振れる次元はそのまま（`arm_flex` は |z| = 62.6 のまま残る）。
