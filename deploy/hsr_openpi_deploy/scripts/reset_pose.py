#!/usr/bin/env python3
import rospy
import yaml
import os
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# 保存ファイルのパス
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), "saved_pose.yaml")

# 関節グループの定義
ARM_JOINTS = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint"
]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
HAND_JOINTS = ["hand_motor_joint"]

def publish_trajectory(pub, joint_names, positions, duration=5.0):
    """指定された関節と位置でJointTrajectoryメッセージを作成して送信"""
    traj = JointTrajectory()
    traj.joint_names = joint_names
    point = JointTrajectoryPoint()
    point.positions = positions
    point.velocities = [0.0] * len(positions) # 速度0で到達
    point.time_from_start = rospy.Duration(duration)
    traj.points = [point]
    pub.publish(traj)

def main():
    rospy.init_node('reset_pose')
    
    # 保存された姿勢ファイルがあるか確認
    if not os.path.exists(SAVE_FILE_PATH):
        rospy.logerr(f"No saved pose found at {SAVE_FILE_PATH}. Please run save_pose.py first.")
        return

    # ファイル読み込み
    try:
        with open(SAVE_FILE_PATH, 'r') as f:
            pose_data = yaml.safe_load(f)
    except Exception as e:
        rospy.logerr(f"Failed to load pose file: {e}")
        return

    # パブリッシャーの準備
    arm_pub = rospy.Publisher("/hsrb/arm_trajectory_controller/command", JointTrajectory, queue_size=1)
    head_pub = rospy.Publisher("/hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=1)
    gripper_pub = rospy.Publisher("/hsrb/gripper_controller/command", JointTrajectory, queue_size=1)

    # 接続待ち
    rospy.loginfo("Waiting for publishers to connect...")
    rospy.sleep(1.0)

    # データの準備
    try:
        arm_positions = [pose_data[j] for j in ARM_JOINTS]
        head_positions = [pose_data[j] for j in HEAD_JOINTS]
        hand_positions = [pose_data[j] for j in HAND_JOINTS]
    except KeyError as e:
        rospy.logerr(f"Missing joint data in saved file: {e}")
        return

    rospy.loginfo("Resetting to saved pose...")
    rospy.loginfo(f"Arm: {arm_positions}")
    rospy.loginfo(f"Head: {head_positions}")
    rospy.loginfo(f"Hand: {hand_positions}")
    
    # コマンド送信
    publish_trajectory(arm_pub, ARM_JOINTS, arm_positions)
    publish_trajectory(head_pub, HEAD_JOINTS, head_positions)
    publish_trajectory(gripper_pub, HAND_JOINTS, hand_positions)

    rospy.loginfo("Commands sent. Waiting for execution...")
    rospy.sleep(5.0) # 動作完了まで待機（簡易的）
    rospy.loginfo("Done.")

if __name__ == "__main__":
    main()
