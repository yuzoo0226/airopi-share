---
name: train-a100
description: Fine-tune pi0.5 on a converted LeRobot dataset using the A100 Slurm cluster. Inspects the dataset, asks about the choices that cannot be inferred from it, builds the config and norm_stats, submits with sbatch, and leaves a watcher running. Use when the user says "train-a100 <dataset path>", or asks to start a training run on the A100 / aquamarine cluster.
---

# train-a100

    /train-a100 <path to a converted LeRobot dataset>

Takes a dataset that has already been converted to LeRobot v2.1 and gets a
training run going on the A100 cluster. The work is mostly *deciding* — the
mechanical parts are three commands. What follows is arranged so the decisions
come after the evidence needed to make them.

**Do not skip to submitting.** Every failure recorded in `docs/a100_training_ja.md`
was silent: the job stayed in RUNNING, the loss looked plausible or was never
printed at all, and the run was wasted. The checks below are each there because
one of them fired.

---

## 1. Read the dataset before asking anything

Run these first. Several of the questions in step 2 answer themselves, and asking
the user something the metadata already states wastes their time.

```bash
D=<dataset path>
python3 -c "
import json; d=json.load(open('$D/meta/info.json'))
print(d['total_episodes'],'episodes /',d['total_frames'],'frames @',d['fps'],'fps')
for k,v in d['features'].items():
    print(f'  {k:28s} {v[\"dtype\"]:8s} {v.get(\"shape\")}')"
cat $D/meta/tasks.jsonl
head -1 $D/meta/episodes.jsonl | python3 -m json.tool | head -40
```

Then three checks that have each caught something real:

**Joint values against the HSR's range.** `episodes_stats.jsonl` carries per-episode
min/max, so this costs nothing and needs no parquet read. Five episodes with
`wrist_flex` at −2.79 rad — 0.6% of frames — put the step-0 loss at 522,160,192
instead of 0.15, and removing them was half of what took closed-loop success from
5% to 35%. Limits are in `deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py`
(`STATE_LIMITS`), margin 0.2.

**Which action columns exist, and whether the base actually moves.** A real-robot
dataset typically carries `action.absolute` (8), `action.state_diff` (8) and
`action.relative` (11 = 8 joints + base_x/base_y/base_t). Pool the per-episode
means and stds from `episodes_stats.jsonl` and look at the base dimensions: if
they are non-zero, `action_mode: relative` with `base_action_dim: 3` is the only
choice that keeps the approach.

**Dimensions whose std is near zero.** Normalisation divides by `std + 1e-6`, so a
joint the task holds still turns controller jitter into enormous normalised
values — a measured 0.0002 gave |z| up to 235, and 235² lands in the loss as a
5e4 spike. Exact zero is safe (the numerator is zero too); small-but-not-zero is
not. Report any dimension under 0.01 as a candidate for the `--min-std` floor.

Also note the **episode count against the name**: `hsr_pens_last50` held 199
episodes, because "last50" counted teleoperation sessions and each was cut into
four primitive actions. Work this out from `source_bag` before asking about it.

---

## 2. Ask about what the dataset cannot tell you

Use `AskUserQuestion`. Ask only what actually changes the run — everything above
is already settled by then. The four that have genuinely needed asking:

**Prompt.** `prompt_from_task: true` uses the episode's own task string. When a
dataset has both primitive actions and a long-horizon label, that choice decides
what the policy can be commanded to do at inference. Note that the base model was
trained over its 75 tasks with `prompt_from_task: true`, so primitive actions
match its distribution; a long-horizon label identical on every episode carries no
information.

**Budget, in epochs.** Compute `frames / batch_size` for steps-per-epoch and
present the options in epochs and wall-clock, not steps. Measured on this cluster:
**2.28 s/step** for synthetic data, **3.8 s/step** for real-camera data (see §6).
The useful range from the Gazebo runs was 2.8 to 8.5 epochs; below about 2 epochs
the model is no better than predicting the dataset mean, which scores well on
open-loop error and cannot do the task at all.

**Which checkpoints survive.** `keep_period` decides this and there is no undoing
it. When there is no automatic closed-loop evaluation — always the case for a real
robot — every kept checkpoint is one the user has to try by hand, so ask how many
they want rather than assuming three.

**Failed episodes.** `task_success` and `success_short_horizon_task` are separate
flags and can disagree. Say how many episodes each would remove before asking.

Also confirm rather than assume, in one line each: the fine-tune recipe (§3), and
whether to match the base model's `ema_decay`.

---

## 3. "action_vision_lora" is the base model's own recipe

There is no preset by that name in this repository. What it means is the recipe
`airoa-pi05-hsr-base` was itself trained with — its `experiment_config.yaml` names
the run `pi05_hsr_75tasks_fast_multinode_8nodes_vision_lora_action_full`:

```yaml
model:
  image_encoder_mode: per_image     # MUST match the checkpoint
  ema_decay: 0.99
  finetune_recipe:
    freeze_text_tower: true         # text tower frozen
    train_action_expert: true       # action expert and head in full
    train_action_head: true
    vision_train_mode: lora         # vision through LoRA
    vision_lora: {enabled: true, rank: 16, alpha: 16.0,
                  targets: [patch_embedding, attention, mlp, head]}
```

Read `checkpoints/airoa-pi05-hsr-base/experiment_config/experiment_config.yaml`
rather than trusting this copy — if the base model is ever replaced, that file is
the authority and this section is stale.

`image_encoder_mode` is the one field that fails silently. `pi0` names the image
encoders `shared` → every key maps to `img`, or `per_image` → the first key maps
to `img` and the rest to `img_<key>`. Loading a `per_image` checkpoint into a
`shared` model keeps only the encoder behind `IMAGE_KEYS[0]` — for HSR the
all-zero, masked-out `base_0_rgb` slot, i.e. essentially untouched PaliGemma
weights — and drops the HSR-adapted hand and head encoders. Nothing errors.

---

## 4. Build it

**Drop the excluded episodes without copying the data.** Rewriting the tree costs a
full copy (tens of GB of PNG inside parquet) and renumbering costs it twice, since
`episode_index` and `index` live inside the parquet files. `filter_lerobot_episodes.py`
edits only the three metadata files and hardlinks the rest, leaving gaps in the
numbering — which `FastLeRobotDataset` supports, because it maps `episode_index` to
position rather than assuming a contiguous range.

```bash
deploy/hsr_openpi_ros2/tools/filter_lerobot_episodes.py \
    <src> datasets/lerobot_datasets/<name> --drop-failed
```

`datasets/lerobot_datasets/` is often owned by root from a container run. Fix it
with `docker exec airopi_ros2_deep_1 chown -R $(id -u):$(id -g) /home/datasets/lerobot_datasets`
rather than with sudo.

**norm_stats**, from the filtered tree, not the original:

```bash
docker exec airopi_ros2_deep_1 bash -lc 'cd /home/openpi && /home/cache/venv/bin/python \
  scripts/aggregate_stats_fast.py \
    --episodes-stats /home/datasets/lerobot_datasets/<name>/meta/episodes_stats.jsonl \
    --chunk-dir      /home/datasets/lerobot_datasets/<name>/data \
    --output-file    assets/<exp>/lerobot_datasets/<name>/norm_stats.json \
    --action-column action.relative --action-mode relative --min-std 0.01'
```

It prints how many stds it raised. Check that against what §1 predicted; a
mismatch means the column or the mode is wrong.

**The config.** Copy `configs/experiments/example_hsr_pens_lora.yaml` — it is the
most recent and carries the reasons in its comments. Name the new one
`example_*.yaml`: `.gitignore` keeps every other `configs/experiments/*.yaml` out
of git, so a different name means the config that produced the run is not recorded
anywhere.

Three fields are not free choices:

- `num_workers: 0`. Above zero, openpi forces the `spawn` start method and a
  spawned worker dies rebuilding its semaphore on this cluster. openpi catches
  that and skips the batch, so the job sits in RUNNING at 95% GPU logging
  "Skipping bad batch" and trains on nothing. This cost job 7899.
- `data_dir: /home/datasets/...` and `params_path: /home/checkpoints/...` —
  container paths. `scripts/a100/train_pick.sh` rewrites both at submit time with
  `sed` anchored on two leading spaces, and then verifies every rewritten path
  exists. Deviate from that spelling and the rewrite silently does nothing.
- `assets_dir: ./assets/<exp>` — relative, and `assets/` is in `.gitignore`, so
  norm_stats never reaches the cluster through git. Transfer it separately.

Validate before shipping:

```bash
docker exec airopi_ros2_deep_1 bash -lc 'cd /home/openpi && /home/cache/venv/bin/python -c "
from openpi.training.experiment_config import ExperimentConfig
print(ExperimentConfig.from_yaml(\"configs/experiments/<name>.yaml\").summary())"'
```

---

## 5. Ship and submit

```bash
ssh yano21@150.69.197.6 'df -h ~ | tail -1'          # 7 TB, but check
rsync -a --delete --info=progress2 datasets/lerobot_datasets/<name> \
    yano21@150.69.197.6:~/usr/airopi/datasets/lerobot_datasets/
rsync -a assets/<exp>/ yano21@150.69.197.6:~/usr/airopi/airopi-share/assets/<exp>/
git push origin devel/ros2 && ssh yano21@150.69.197.6 \
    'cd ~/usr/airopi/airopi-share && git pull --ff-only'
```

`--delete` on the dataset: rebuilding a dataset can reduce the episode count, and
without it the old files stay as orphans the loader will happily read.

Verify on the far side that meta and files agree before submitting — a truncated
rsync looks like a smaller dataset, not like an error.

**Always `sbatch`.** Never `bash train_pick.sh` on the login node; that takes GPUs
outside the scheduler.

```bash
ssh yano21@150.69.197.6 'cd ~/usr/airopi/airopi-share && sbatch -p part_80gb \
  --export=ALL,EXP_NAME=<exp>,CONFIG_YAML=configs/experiments/<name>.yaml \
  scripts/a100/train_pick.sh'
```

Partitions are `part_20gb` / `part_40gb` / `part_80gb`; pi05 at batch 16 needs
`part_80gb`.

---

## 6. Watch it, and read step 0 before doing anything else

Two watchers, always — the user has asked for this explicitly and standing.

```bash
scripts/a100/watch_slurm_job.sh <job_id> 120      # run in background
```

and a `Monitor` on the job's `.out` file. Filter to **one event per 1000 steps**
plus every failure signature — at 50-step granularity a 20-hour run produces 400
notifications and a real failure is lost among them:

    Step [0-9]+000:|Traceback|CUDA_ERROR|out of memory|Skipping bad batch|RESOURCE_EXHAUSTED|Killed

Judge health by **output progress, not process liveness**. `pgrep` through
`docker exec` produced both false positives and false negatives; every monitoring
false alarm in this project came from asking whether a process existed instead of
whether it was still producing output.

**Then read step 0.** It is the single most informative line in the run:

| field | healthy | what a bad value means |
| --- | --- | --- |
| `loss` | 0.05 – 0.2 | 1e8 means joint values or normalisation are broken. Stop. |
| `param_norm` | identical to a previous run from the same base | a different value means the base weights did not load as expected — check `image_encoder_mode` |
| `grad_norm` | order 1 | hundreds means the same thing as a huge loss |
| `data_time_s` vs `step_time_s` | — | see below |

The loss only appears at all because `train_pick.sh` exports `PYTHONUNBUFFERED=1`:
openpi writes it with `pbar.write()`, which bypasses logging and sits in a
block-buffered stdout under Slurm. Without it the run shows tqdm progress for
hours and not one loss value.

**On throughput.** `data_time_s` will usually exceed `step_time_s` by an order of
magnitude, because `num_workers: 0` puts PNG decoding in the main process. This is
the accepted cost of not risking the silent `spawn` failure. Measured:

    synthetic (Gazebo)   129 KB/frame   gpu 0.148 s   data 0.70 s   2.28 s/step
    real camera          616 KB/frame   gpu 0.194 s   data 2.04 s   3.79 s/step

Real camera frames carry sensor noise and texture, so PNG barely compresses them;
synthetic renders are mostly flat surfaces. Budget real-robot runs at roughly
**1.7× the wall clock** of a synthetic run of the same length, and quote the
estimate from a measured rate rather than from a previous run's.

---

## 7. What the numbers will not tell you

Report progress honestly, and resist the pull of a falling validation curve.

**`val_loss` and `action_l1` do not predict closed-loop success.** Measured three
times on the same project. At 20k steps both improved while the task collapsed
from 35% to 5%. At 29k `val_loss` crossed the overwatch threshold that had been
fixed in advance — its only firing in the whole run — on the checkpoint that then
scored 30%, and `action_l1` reached its best value of the run on that same
checkpoint. Both are frame-averaged over held-out episodes; a manipulation task is
decided by the handful of frames where the gripper closes.

Log them. Do not decide on them, and do not tell the user a run is going well
because they are falling.

**Do not stop at a bad checkpoint.** 35% at 10k and 5% at 20k reads as a peak
followed by decline; the 30k point made it a dip. Stopping at 20k would have
produced a confident and wrong conclusion. Let the learning rate finish decaying
before drawing a curve.

**Epochs, not steps.** Two runs matched on steps are matched on compute, not on how
often the model saw its data. 966 steps of 7,726 frames is two passes; 1,000 steps
of 26,042 frames is six tenths of one, and the second model had not finished
looking at its dataset once. When comparing runs, convert to epochs first.

When asked whether training is going well, the accurate answer before ~2 epochs is
that it is not broken.
