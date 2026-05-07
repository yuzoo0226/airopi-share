#!/usr/bin/env python3
import os
import json

import rospy
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from hsr_data_msgs.srv import StringTrigger, StringTriggerResponse


class DualshockInterface:
    def __init__(self):
        rospy.init_node('dualshock_teleop_interface', anonymous=True)

        # parameters
        self.interface = rospy.get_param('dualshock_interface', 'ps3')
        if self.interface == 'ps4':
            self.buttons = rospy.get_param('ps4_scheme', None)
        elif self.interface == 'ps3':
            self.buttons = rospy.get_param('ps3_scheme', None)
        else:
            self.buttons = None
        if rospy.get_param('~use_spacemouse', False):
            self.interface = "spacemouse"
        self.hsr_id = rospy.get_param('hsr_id', 'hsrb_000')
        self.location_name = rospy.get_param('location_name', 'location')
        self.instruction = rospy.get_param('~init_instruction', '')
        self.hash_value = os.getenv('GIT_HASH')
        self.branch_name = os.getenv('GIT_BRANCH')
        self.task_name = self.get_task_name()
        self.arm_joint_names = rospy.get_param('arm_joint_names', [])
        self.head_joint_names = rospy.get_param('head_joint_names', [])
        self.target_arm_poses = [
            rospy.get_param('top_down_joint_poses', {}),
            rospy.get_param('pick_up_joint_poses', {}),
            rospy.get_param('home_joint_poses', {}),
        ]

        self.joint_state = None
        self.is_recording = None
        self.current_mode = "manual"
        self.target_pose_idx = 0
        self.last_button_press_time = rospy.Time.now()
        self.debounce_time = rospy.Duration(1.0)
        self.rate = rospy.Rate(10)

        # publisher
        self.control_mode_pub = rospy.Publisher('/control_mode', String, queue_size=10)
        self.arm_pub = rospy.Publisher("/hsrb/arm_trajectory_controller/command", JointTrajectory, queue_size=10)
        self.head_pub = rospy.Publisher("/hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=10)

        # subscriber
        rospy.Subscriber('/hsrb/joy', Joy, self.joy_callback)
        rospy.Subscriber("/hsrb/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber('/rosbag_manager/recording', Bool, self.rosbag_status_callback)

        # service
        rospy.Service('/update_instruction', StringTrigger, self.update_instruction_srv)
        self.start_recording = rospy.ServiceProxy('/rosbag_manager/start_recording', StringTrigger)
        self.stop_recording = rospy.ServiceProxy('/rosbag_manager/stop_recording', StringTrigger)
        self.delete_recording = rospy.ServiceProxy('/rosbag_manager/delete_recording', Trigger)

    def joint_state_callback(self, data):
        self.joint_state = data

    def rosbag_status_callback(self, data):
        self.is_recording = data.data

    def joy_callback(self, data):
        if rospy.Time.now() - self.last_button_press_time < self.debounce_time:
            return
        if self.is_recording is None:
            return 
        if self.joint_state is None:
            return
        # Toggle rosbag record start/stop
        if data.buttons[self.buttons["PS_DIR_UP"]] == 1:
            self.last_button_press_time = rospy.Time.now()
            if not self.is_recording:
                if self.instruction == "":
                    rospy.logwarn("Please set language instruction.")
                    return
                self.start_recording(self.get_task_name())
            else:
                meta_data = dict(
                    interface=self.interface,
                    instructions=[[self.instruction]],
                    segments=[
                        {
                            "start_time": -1,
                            "end_time": -1,
                            "instructions_index": 0,
                            "has_suboptimal": False,
                            "is_directed": True,
                        }
                    ],
                    interface_git_hash=self.hash_value,
                    interface_git_branch=self.branch_name
                )
                message = json.dumps(meta_data)
                self.stop_recording(message)
        # Delete recent rosbag data
        elif data.buttons[self.buttons["PS_DIR_DOWN"]] == 1:
            self.last_button_press_time = rospy.Time.now()
            self.delete_recording()
        # Change control mode
        elif data.buttons[self.buttons["PS_DIR_LEFT"]] == 1:
            self.last_button_press_time = rospy.Time.now()
            if self.current_mode == "manual":
                self.current_mode = "auto"
            else:
                self.current_mode = "manual"
            rospy.loginfo(f"Control mode changed to: {self.current_mode}")
        # Change Joint State
        elif data.buttons[self.buttons["PS_DIR_RIGHT"]] == 1:
            self.last_button_press_time = rospy.Time.now()
            if self.current_mode == "manual" and not self.is_recording:
                rospy.loginfo(f"Change joint state: {self.target_pose_idx}")
                self.change_joint_state(self.target_arm_poses[self.target_pose_idx])
                if self.target_pose_idx == len(self.target_arm_poses)-1:
                    self.target_pose_idx=0
                else:
                    self.target_pose_idx+=1
    
    def update_instruction_srv(self, req):
        self.instruction = req.message
        self.task_name = self.get_task_name()
        rospy.loginfo(f'Instruction is updated to "{self.instruction}".')
        res = True
        return StringTriggerResponse(success=res)

    def get_task_name(self):
        task_name = self.instruction.lower() + "_" +self.hsr_id + "_" + self.location_name + "_" + self.interface
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            task_name = task_name.replace(char, '')
        replace_chars = ' ,.'
        for char in replace_chars:
            task_name = task_name.replace(char, '_')
        if task_name.endswith('_'):
            task_name = task_name[:-1]
        return task_name

    def change_joint_state(self, target_pose):
        # arm
        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names
        p = JointTrajectoryPoint()
        p.positions = []
        for joint_name in self.arm_joint_names:
            pose = self.joint_state.position[self.joint_state.name.index(joint_name)]
            for target_joint, target_value in target_pose.items():
                if target_joint == joint_name:
                    pose = target_value
                    break
            p.positions.append(pose)
        p.velocities = []
        p.time_from_start = rospy.Duration(0.1)
        traj.points = [p]
        self.arm_pub.publish(traj)
        # head
        traj = JointTrajectory()
        traj.joint_names = self.head_joint_names
        p = JointTrajectoryPoint()
        p.positions = []
        for joint_name in self.head_joint_names:
            pose = self.joint_state.position[self.joint_state.name.index(joint_name)]
            for target_joint, target_value in target_pose.items():
                if target_joint == joint_name:
                    pose = target_value
                    break
            p.positions.append(pose)
        p.velocities = []
        p.time_from_start = rospy.Duration(0.1)
        traj.points = [p]
        self.head_pub.publish(traj)

    def process(self):
        while not rospy.is_shutdown():
            self.control_mode_pub.publish(self.current_mode)
            self.rate.sleep()

        if self.is_recording:
            self.delete_recording()

def main():
    dualshock_interface = DualshockInterface()
    dualshock_interface.process()

if __name__ == '__main__':
    main()
