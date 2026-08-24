# 実機テレオペとデータ収録（ROS 2）

`hsr_leader_teleop`（リーダーアーム + JoyCon / キーボード）で HSR をテレオペしながら、
**HSR 本体に ROS 2 bag を保存**し、そのまま学習データに変換するための手順です。

* 変換・学習: [`ros2_data_collection_ja.md`](ros2_data_collection_ja.md)
* 推論: [`ros2_deploy_ja.md`](ros2_deploy_ja.md)

---

## 1. なぜそのまま繋がるのか

[`Hibikino-Musashi-Home/hsr_leader_teleop`](https://github.com/Hibikino-Musashi-Home/hsr_leader_teleop)
は既に ROS 2 Humble 実装で、指令トピックが本デプロイと**完全に一致**しています。

| | hsr_leader_teleop | 本リポジトリ (hsr_openpi_node / pick_task) |
| --- | --- | --- |
| アーム | `/arm_trajectory_controller/joint_trajectory` | 同じ |
| ヘッド | `/head_trajectory_controller/joint_trajectory` | 同じ |
| グリッパ | `/gripper_controller/joint_trajectory` | 同じ |
| 把持 | `/gripper_controller/grasp` (`tmc_control_msgs/GripperApplyEffort`) | 同じ |
| 台車 | `/omni_base_controller/cmd_vel` | 同じ |
| 関節角 | `/joint_states` | 同じ |

したがって、テレオペ中に上記を録った bag は
`deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py` にそのまま食わせられます。

> 実機では上記が `/hsrb/` 名前空間に入ります。レコーダの `profile:=real` が
> その組を、`profile:=sim` がシミュレータの組を選びます。

---

## 2. テレオペ側の準備

```bash
git clone https://github.com/Hibikino-Musashi-Home/hsr_leader_teleop.git
cd hsr_leader_teleop
docker build . -t docker.hsr.leader_teleop:humble
sudo cp 99-hsr-leader-udev.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
./RUN-DOCKER-CONTAINER.sh
# コンテナ内
ros2 launch hsr_leader_teleop hsr_leader_teleop.launch.py
```

* リーダーアームは `/dev/ttyHSR_LEADER`（udev ルールで固定）に Dynamixel で接続します。
* JoyCon を使わない場合、`hsr_leader_teleop.py` は **キーボード操作**にも対応しています
  （`feature/add-keyboard-control` ブランチ、`keyboard` パッケージを使うため root 権限が必要）。
  launch では joycon ノードは既定でコメントアウトされています。
* DDS は CycloneDDS + `cyclonedds_profile.xml`、`ROS_DOMAIN_ID` は
  `RUN-DOCKER-CONTAINER.sh` の `HSR_DOMAIN_ID`（既定 54）です。
  **収録ノードも同じ DOMAIN_ID / RMW に揃えてください。**

---

## 3. HSR 本体での収録

収録ノードは `hsr_openpi` パッケージに入っています。HSR 本体（またはロボットと
同じ DDS ドメインにいる機器）で実行してください。bag は実行したマシンに書かれます。

```bash
ros2 launch hsr_openpi teleop_record.launch.py \
    profile:=real \
    output_dir:=/home/administrator/hsr_bags \
    task:="pick up the bottle"
```

エピソード単位の制御:

```bash
# 開始（メッセージがそのままタスク文字列 = 学習時の prompt になります）
ros2 service call /hsr_bag_recorder/start_episode \
    hsr_openpi_msgs/srv/StringTrigger "{message: 'pick up the bottle'}"

# 終了（成功した試行）
ros2 service call /hsr_bag_recorder/stop_episode std_srvs/srv/Trigger

# 破棄（失敗した試行 — end ではなく discard が記録され、変換時に無視されます）
ros2 service call /hsr_bag_recorder/discard_episode std_srvs/srv/Trigger
```

出力:

```
/home/administrator/hsr_bags/
├── teleop_20260824_181500/          # bag 本体（mcap）
│   ├── metadata.yaml
│   └── teleop_20260824_181500_0.mcap
└── teleop_20260824_181500.episodes.json   # エピソード一覧（index / task / 長さ）
```

### 3.1 主なパラメータ

| 引数 | 既定 | 意味 |
| --- | --- | --- |
| `profile` | `real` | `real` (`/hsrb/...`) / `sim` / `custom`（`topics` で指定） |
| `output_dir` | `~/hsr_bags` | 保存先（**このノードを動かしたマシン**上） |
| `bag_name` | `""` | 空ならタイムスタンプ |
| `storage` | `mcap` | `mcap` / `sqlite3` |
| `compression` | `""` | `zstd` を指定すると圧縮（CPU と引き換えに容量削減） |
| `max_bagfile_size` | `0` | N バイトで分割（0 = 単一ファイル） |
| `min_episode_seconds` | `1.0` | これより短いエピソードは自動で破棄 |
| `auto_start` | `false` | 起動と同時に 1 本目を開始 |

### 3.2 重要: 終了は必ず Ctrl-C で

rosbag2 は **SIGINT を受けたときにだけ** `metadata.yaml` を書きます。
`kill -9` すると `ros2 bag info` も変換スクリプトも開けない bag になります。
本ノードは終了時に `ros2 bag record` へ SIGINT を送るので、**ノードを Ctrl-C で
止めてください**（強制終了しないこと）。

壊れてしまった bag は次で復旧できます。

```bash
ros2 bag reindex <bag_dir> -s mcap        # metadata.yaml を再生成
# それでも rosbags が "File end magic is invalid" と言う場合は書き直す
ros2 bag convert -i <bag_dir> -o convert.yaml
```

---

## 4. 実機に触る前にシミュレータで通す

トピック名以外は同じなので、Gazebo で一連の流れを確認できます。

```bash
# 端末1: シミュレータ
ros2 launch hsr_openpi hsr_sim.launch.py world:=pick_table

# 端末2: 収録
ros2 launch hsr_openpi teleop_record.launch.py profile:=sim use_sim_time:=true \
    output_dir:=/home/hsr/hsr_ros2_ws/_bags

# 端末3: 何かでロボットを動かす（テレオペ、あるいは pick_task）
ros2 run hsr_openpi pick_task --ros-args -p use_sim_time:=true -p num_episodes:=3
```

---

## 5. 収録後: 学習データへ

```bash
# 学習コンテナ側
uv pip install --python /home/cache/venv/bin/python rosbags   # 初回のみ
/home/cache/venv/bin/python deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py \
    --bag /home/bags/teleop_20260824_181500 \
    --repo-id lerobot_datasets/hsr_teleop \
    --root /home/datasets --fps 10 --image-order bgr \
    --topics-json deploy/hsr_openpi_ros2/config/real_bag_topics.json
```

実機 bag は `/hsrb/` 名前空間なので、`--topics-json` でトピック名を渡します
（`config/real_bag_topics.json` を同梱）。

動画で中身を確認する場合:

```bash
python deploy/hsr_openpi_ros2/tools/bag2video.py \
    --bag /home/bags/teleop_20260824_181500 --out teleop.mp4 --episodes 0,1,2
```

---

## 6. 収録トピック一覧

`profile:=real` で録るもの（`bag_recorder.py` の `REAL_ROBOT_TOPICS`）:

```
/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed
/hsrb/head_rgbd_sensor/rgb/camera_info
/hsrb/hand_camera/image_raw/compressed
/hsrb/hand_camera/camera_info
/hsrb/joint_states
/hsrb/arm_trajectory_controller/joint_trajectory
/hsrb/head_trajectory_controller/joint_trajectory
/hsrb/gripper_controller/joint_trajectory
/hsrb/omni_base_controller/cmd_vel
/hsrb/omni_base_controller/wheel_odom
/hsrb/wrist_wrench/raw
/tf  /tf_static  /control_mode
```

ROS 1 のデータ収集で録っていた組（`/hsrb/command_velocity`,
`/hsrb/gripper_controller/grasp/goal` など）との違いは名前空間内の
コントローラ名です。ROS 2 の `joint_trajectory_controller` は
`~/command` ではなく `~/joint_trajectory` を購読します。

> **把持アクションについて**: ROS 2 のアクションはトピックとして録れないため
> （`/gripper_controller/grasp` はサービス + フィードバック）、把持は
> `/gripper_controller/joint_trajectory` の連続値として記録されます。
> テレオペ側が力制御の grasp アクションだけで閉じる運用の場合は、
> `hsr_leader_teleop` 側で開度指令も publish するようにしてください
> （そうしないと学習データのグリッパ次元が動きません）。
