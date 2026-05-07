# HSR Data Collection

## 本コードの依存関係
- ubuntu >= 20.04
- docker
- nvidia-docker

## 初期設定

1. コードのダウンロード
    ```bash
    git clone https://git.hsr.io/hsrtx/hsr_data_collection.git
    ```

2. Dockerイメージのビルド
    ```bash
    cd hsr_data_collection
    docker build . -f docker/Dockerfile -t hsr_data_collection:noetic-cuda11
    ```

3. ネットワーク設定
    - [基本的なネットワーク設定](https://docs.hsr.io/hsrb_user_manual/howto/network_settings.html#id3)の場合

    `echo "export ROBOT_NAME=hsrb100" >> ${HOME}/.bashrc`
    
    - 固定IPで運用している場合

    `echo "export HSR_IP=192.168.0.100" >> ${HOME}/.bashrc`

4. データセットパス設定(任意)
    - 収集データの保存先を指定したい場合は、下記の変数を設定してください。
    `export HSR_DATASET_DIR=/path/to/your/dataset`

    - 設定しない場合は `hsr_data_collection/datasets`にデータが保存されます。 

5. ROSパッケージのビルド

    ```bash
    ./RUN-DOCKER-CONTAINER.sh
    roscd
    catkin build
    ```

## データ収集
### 初期設定

1. configの設定

    `hsr_data_collection/config/hsr_data_collection_config.yaml`の下記の値を変更してください。

    - hsr_id: hsrのIDを記載してください。(e.g. hsrb_100, hsrc_050)
    - location_name: (e.g. tmc, u_tokyo)
    - dualshock_interface: ["ps3", "ps4"] から選択してください。

2. HSRのteleop操作方法の変更

    本データ収集コードでは操作方法はパターン2を前提にしています。
    
    [HSRC Dualshock 操作方法](https://docs.hsr.io/hsrc_user_manual/howto/use_dualshock_controller.html#change-dualshock-control-pattern-label) を参考に、USE_JOYSTIC_TELEOP2にしてください。

### Dualshock操縦によるデータ収集
1. 起動方法

    ```bash
    roslaunch hsr_data_collection hsr_teleop_data_collection.launch
    ```

2. 操縦方法
    - 基本操作：通常のHSR操縦方法(パターン2)と基本的に同じです。

    - 基本姿勢遷移：右ボタン押下で、事前に複数設定してある姿勢に遷移します。
        
        （rosbag recording中は遷移しません）

3. 録画開始・保存方法

    - 上ボタン押下：rosbagが録画中の場合は、停止し保存。それ以外の場合は録画開始。
    - 下ボタン押下：録画中もしくは直近保存したrosbagの削除。

4. 言語指示の更新方法

    ```bash
    rosservice call /update_instruction "message: 'Pick up a banana on the table.'"
    ```

5. ステータスLED表示
    
    - 紫色："auto"モード
    - 青緑色: "manual"モード
    - 点滅: rosbagレコード中

    ※ 操縦モード：左ボタン押下で"manual"と"auto"が変更されます。
    
    遠隔操作でのデータ収集時は、manualモードにしてください。

### 3D spacemouse操縦によるデータ収集
1. 起動方法

    ```bash
    spacenavd
    roslaunch hsr_data_collection hsr_teleop_data_collection.launch use_spacemouse:=true
    ```

2. 操縦方法

    dualshockの右ボタン押下で、トップダウンのピック姿勢に遷移して使うことを前提にしています。

    3Dマウスでの操作方法は下記の通り。

    - X方向: ベース横移動 (右ボタン押下時は、速度UP)
    - Y方向: ベース前後移動 (右ボタン押下時は、速度UP)
    - Z方向: アーム上下移動
    - yaw方向: リストroll方向回転 (右ボタン押下時は、ベースyaw方向回転)
    - 左ボタン: グリッパー開閉

    上記以外の操作は、dualshokでの操作でも可能。

    録画の開始・保存、言語指示の変更は、dualshock時と同じ。

### Quest 2/3によるデータ収集

1. VR teleop (名古屋大学長谷川研提供版)
   - **このVR teleopを使用して研究を進め、論文を執筆する場合は、以下の引用をお願いします。**
      ```
      [1] Nakanishi, J., Itadera, S., Aoyama, T., & Hasegawa, Y. (2020). Towards the development of an intuitive teleoperation system for human support robot using a VR device. Advanced Robotics, 34(19), 1239–1253.
      ```
    - **HSRTX以外の用途で使用する場合は、名古屋大学長谷川先生([hasegawa@mein.nagoya-u.ac.jp](mailto:hasegawa@mein.nagoya-u.ac.jp))の同意を得てからご使用ください。**


   0. 事前準備
      - awsからapkファイルととdocker image(tarファイル)をダウンロード
        - aws s3 cp s3://dev-frc-hsr-data-bucket/nir_teleop/ /path/to/download/
        - docker load < nir_hsrtx_img.tar
      - QuestをPCと繋げてawsからダウンロードしたapkをSideQuestでインストール
      - wifiと有線LANが使えるUbuntuPCを用意
      - UbuntuPCをHSRと接続できるネットワークのルータもしくはスイッチングハブと有線LANで接続
      - UbuntuPCでwifiのhotspotをON
      - Questをhotspotのwifiに接続
      - hsr_data_collection/scripts/nir_vr_teleopにあるdocker-compose.yamlを以下のように編集
        - ROS_MASTER_URI=http://{hsrのIP}:11311
        - ROS_IP={UbuntuPCのIP}
   1. 起動方法
      - Questでアプリを実行
      - docker外でnir_quest_teleopノードの起動
        ```bash
        cd hsr_data_collection/scripts/nir_vr_teleop
        docker compose up
        ```
      - docker内でhsr_data_collectionのノードを起動
        ```bash
        roslaunch hsr_data_collection hsr_nir_vr_teleop_data_collection.launch
        ```
      - また、これから収集する動作の言語指示（ラベル）をdualshockと同様にrosserviceで設定

    2. 操縦方法
        - キャリブレーション
            音声案内に従い、遠隔操作に使いたいコントローラのAボタンもしくはXボタンを長押しする。
        - 遠隔操作するコントローラ
            - gripボタン: 遠隔操作制御トリガー
            - A/Xボタン: gripボタン押下時に、左旋回 (A/Xボタンのみ、長押しするとIKモードの切り替え)
            - B/Yボタン: gripボタン押下時に、右旋回
            - joy: gripボタン押下時、台車の左右前後方向移動
            - trigger: gripボタン押下時に、グリッパー開閉(5秒以上押し続けるとホールドします。再度押下すると解除。)
        - 遠隔操作をしていないコントローラ
            - gripボタン: gripper開閉
            - A/Xボタン: rosbagが録画中の場合は、停止し保存。それ以外の場合は録画開始。
            - X/Yボタン: 録画中もしくは直近保存したrosbagの削除。
            - joy: -
            - trigger: control modeの切り替え
            - joyボタン: 再キャリブレーション(左右のコントローラ変更することも可能)
        
2. VR teleop (TMC版)

    1. 起動方法
        - ALVRを起動し、QuestとPCを接続
        - 下記のlaunchファイルを起動
        ```bash
        roslaunch hsr_data_collection hsr_vr_teleop_data_collection.launch
        ```
        - 下記のスクリプトを実行(local環境)
        ```bash
        cd hsr_data_collection/scripts/vr_system
        python3 vr_sysyte.mpy
        ```

        ※ openvrライブラリをインストールする必要があります。
        
        `pip3 install openvr`
    
    2. 操縦方法
        - 左コントローラ
            - Xボタン: rosbagが録画中の場合は、停止し保存。それ以外の場合は録画開始。
            - Yボタン: 録画中もしくは直近保存したrosbagの削除。
            - joy: 左gripボタンを押下時、台車の前後左右方向移動
            - gripボタン: 台車移動トリガー
            - trigger: 速度UP
        - 右コントローラ
            - Aボタン: 基本姿勢遷移
            - Bボタン: コントロールモード変更
            - joy: 左gripボタン押下時、台車の回転方向移動
            - gripボタン: 手先制御トリガー
            - trigger: 手先制御時の台車移動ON

### Quest 2/3による遠隔地テレオペレーション
VPNを使った遠隔地テレオペレーションの方法になります。名大のVR teleopを使用して、遠隔地テレオペレーションを行うことができます。

1. AWSでEC2のサーバーを建てる
    - AWSのアカウントにログインして、EC2のdashboardに移動し、"Launch Instance"をクリック。
    - Ubuntuのインスタンスを選択し、VPNで使用されるポート（例: port 51820）へのインバウンドおよびアウトバウンドトラフィックを許可するようにセキュリティグループを設定します。
2. AWS乗にVPNサーバーを準備する
    - 以下のチュートリアルを参考に、VPNサーバーを構築します。 https://github.com/DavidYaonanZhu/wireguard-install
3. VPNクライアントをHSRと操作側のPCで設定する
    - VPNクライアントの設定は以下のリンクを参考にしてください。 https://github.com/DavidYaonanZhu/wireguard-install
    - VPNクライアントをHSRと操作側のPCで設定します。
    - HSRにSSHして、以下のコマンドを実行してください。
        ```bash
        wg-quick up follower
        ```
    - 操作側のPCで、以下のコマンドを実行してください。
        ```bash
        wg-quick up leader
        ```
    - VPNを起動した後に、ifconfig等でIPアドレスを確認してください。IPアドレスが10.x.x.xになっていることを確認してください。
    - 実験が終わったら、以下のコマンドを実行してください。
        ```bash
        wg-quick down follower
        ```
        ```bash
        wg-quick down leader
        ```
4. IPアドレス、ROS_MASTER_URIを設定して遠隔操作のノードを起動する。
    - HSR、操作側のPC両方で、ROS_IPをVPNのIPアドレスに設定してください。
        - HSR側は、/etc/opt/tmc/robot/docker.hsrb.userの中でROS_IPを設定してください。
        - もし、以上でも接続できない場合は、操作側のPCでROS_MASTER_URI、ROS_HOSTNAMEも設定してください。
        - HSRのROSノードが操作側のPCから見えるようになったら、teleoperationのnodeを起動してください。

Demo Video
https://youtu.be/VjqJRzn4ud8

Author: Yaonan Zhu
Affiliation:  Assistant Professor, The University of Tokyo
Contact: yaonan.zhu@weblab.t.u-tokyo.ac.jp
Reference:
Consider citing the following work
```
Y. Zhu, B. Jiang, Q. Chen, T. Aoyama and Y. Hasegawa, "A Shared Control Framework for Enhanced Grasping Performance in Teleoperation," in IEEE Access, vol. 11, pp. 69204-69215, 2023, doi: 10.1109/ACCESS.2023.3292410.
```

<!-- Teleoperation from anywhere through VPN
Author: Yaonan Zhu
Affiliation:  Assistant Professor, The University of Tokyo
Contact: yaonan.zhu@weblab.t.u-tokyo.ac.jp

1. Apply AWS account, start up EC2 server
    1. Log into your AWS account.
    2. Navigate to the EC2 dashboard and click on “Launch Instance.”
    3. Select a Linux-based instance (Ubuntu).
    4. Configure security groups to allow inbound and outbound traffic on ports used by VPN (e.g. port 51820).
2. Setup VPN server on AWS
    1. Follow the tutorial link https://github.com/DavidYaonanZhu/wireguard-install
3. Setup VPN client on HSR and Leader PC
    1. Follow the tutorial link about how to setup VPN clients https://github.com/DavidYaonanZhu/wireguard-install
    2. Startup VPN clients on HSR and the Leader PC
    3. SSH into HSR
        wg-quick up follower
    4. On the leader PC
        wg-quick up leader
    5. Check ip addresses of both machines: ifconfig should give you ip address starting from 10.x.x.x
    6. If finished the experiment, do not forget to shutdown the VPN client wg-quick down xxx
4. Setting IP addresses, ROS_MASTER_URI, Start teleoperation nodes
    1. On both machines, ROS_MASTER_URI should be set to the VPN address 10.x.x.HSR (check this on your HSR)
    2. Also set ROS_IP, ROS_HOSTNAME accordingly to the VPN client addresses
    3. Run the teleoperation nodes
    
Demo Video
https://youtu.be/VjqJRzn4ud8
Reference
Consider citing the following work

Y. Zhu, B. Jiang, Q. Chen, T. Aoyama and Y. Hasegawa, "A Shared Control Framework for Enhanced Grasping Performance in Teleoperation," in IEEE Access, vol. 11, pp. 69204-69215, 2023, doi: 10.1109/ACCESS.2023.3292410. -->

### 自作コードによるデータ収集

hsr_data_collectionのrosbag_manager.py Nodeを起動することで、
下記のROS Serviceを使用し、遠隔操作と同様のmeta.jsonの保存とrosbagの操作が可能です。

- `/rosbag_manager/start_recording`
    - サービス型：hsr_data_msgs/StringTrigger
    - 引数
        - message: タスク名 (例: tidyup_202410011312_hsrc_054_matsuolab)
    - 返り値
        - success: 録画開始が成功の場合、true
- `/rosbag_manager/stop_recording`: 
    - サービス型：hsr_data_msgs/StringTrigger
    - 引数
        - message: metaデータ
    - 返り値
        - success: 録画停止とmetaデータ保存の成功の場合、true
    - metaデータ作成の例

        `initialize_meta_data`で初期化し、`record_function_call_time`で起動ごとにmetaデータを更新。

        json.dumpsでstr型にしてservice実行。

        ```python
            def initialize_meta_data(self):
            meta_data = dict(
                interface="tidyup",
                instructions=[],
                segments=[],
                interface_git_hash=self.hash_value,
                interface_git_branch=self.branch_name,
            )
            return meta_data

            def record_function_call_time(self, instruction, start_time, end_time, is_successed):
                self.meta_data["instructions"].append([instruction])
                self.meta_data["segments"].append(
                    dict(
                        start_time=start_time,
                        end_time=end_time,
                        instructions_index=len(self.meta_data["segments"]),
                        has_suboptimal=is_successed,
                        is_directed=True
                    )
                )

        meta_data_str = json.dumps(self.meta_data)
        self.stop_recording(meta_data_str)
        ```


## データ変換
### rosbag -> rlds 変換

```bash
cd conversion
python3 rosbag2pkl.py /path/to/dataset/rosbag_data /path/to/dataset/pkls
python3 pkl2rlds.py /path/to/dataset/pkls /path/to/dataset/rlds_dataset
```
tf_bagのimportエラーが出る場合は、source /root/external_catkin_ws/devel/setup.bashしてから実行してください。

## データアップロード

```bash

aws s3 sync /path/to/dataset/ s3://dev-frc-hsr-data-bucket/AAA
```
