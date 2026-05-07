#!/usr/bin/env python3
import rospy
import yaml
import os
from sensor_msgs.msg import JointState

# 保存ファイルのパス（このスクリプトと同じディレクトリに保存）
SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), "saved_pose.yaml")

# 保存対象の関節名
TARGET_JOINTS = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "hand_motor_joint"
]

def callback(msg):
    pose_data = {}
    # メッセージから対象の関節角度を抽出
    for name in TARGET_JOINTS:
        if name in msg.name:
            idx = msg.name.index(name)
            pose_data[name] = float(msg.position[idx])
    
    # 全ての関節が見つかったか確認（念のため）
    if len(pose_data) < len(TARGET_JOINTS):
        rospy.logwarn("Not all joints found in the message yet...")
        return

    # ファイルに保存
    try:
        with open(SAVE_FILE_PATH, 'w') as f:
            yaml.dump(pose_data, f)
        
        rospy.loginfo(f"Current pose saved to: {SAVE_FILE_PATH}")
        rospy.loginfo(f"Data: {pose_data}")
        
        # 保存できたらノードを終了
        rospy.signal_shutdown("Saved successfully")
    except Exception as e:
        rospy.logerr(f"Failed to save pose: {e}")

def main():
    rospy.init_node('save_pose')
    rospy.loginfo("Waiting for joint states...")
    
    # JointStateを購読。メッセージが来たらcallbackを実行
    sub = rospy.Subscriber("/hsrb/joint_states", JointState, callback)
    
    rospy.spin()

if __name__ == "__main__":
    main()
