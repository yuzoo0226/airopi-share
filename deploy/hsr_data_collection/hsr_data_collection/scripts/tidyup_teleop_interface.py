#!/usr/bin/env python3
import os
from datetime import datetime, timezone, timedelta
import json
import time

import numpy as np
import rospy
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tmc_msgs.msg import TalkRequestAction, TalkRequestGoal, Voice
from hsr_data_msgs.srv import StringTrigger

from command_generator import CommandGenerator


class TidyupTeleopInterface:
    def __init__(self):
        rospy.init_node('tidyup_teleop_interface', anonymous=True)

        # parameters
        self.interface = rospy.get_param('dualshock_interface', 'ps3')
        if self.interface == 'ps4':
            self.buttons = rospy.get_param("ps4_scheme", None)
        elif self.interface == 'ps3':
            self.buttons = rospy.get_param("ps3_scheme", None)
        else:
            self.buttons = None
        self.hsr_id = rospy.get_param('hsr_id', 'hsrb_000')
        self.location_name = rospy.get_param('location_name', 'location')
        self.instruction = rospy.get_param('~init_instruction', '')
        self.hash_value = os.getenv('GIT_HASH')
        self.branch_name = os.getenv('GIT_BRANCH')
        location_names = rospy.get_param('location_names', ['on the floor'])
        object_names = rospy.get_param('object_names', ['apple'])
        self.arm_joint_names = rospy.get_param('arm_joint_names', [])
        self.head_joint_names = rospy.get_param('head_joint_names', [])
        self.target_arm_poses = [
            rospy.get_param('top_down_joint_poses', {}),
            rospy.get_param('pick_up_joint_poses', {}),
            rospy.get_param('home_joint_poses', {}),
        ]
        self.command_generator = CommandGenerator(location_names, object_names)
        self.joint_state = None
        self.is_recording = None
        self.start_time = None
        self.target_pose_idx = 0
        self.status = -1
        self.has_object = False
        self.object_name = None
        self.meta_data = self.initialize_meta_data()
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
        self.start_recording = rospy.ServiceProxy('/rosbag_manager/start_recording', StringTrigger)
        self.stop_recording = rospy.ServiceProxy('/rosbag_manager/stop_recording', StringTrigger)
        self.talk_request_client = SimpleActionClient('/talk_request_action', TalkRequestAction)

    def joint_state_callback(self, data):
        self.joint_state = data

    def rosbag_status_callback(self, data):
        self.is_recording = data.data

    def joy_callback(self, data):
        if rospy.Time.now() - self.last_button_press_time < self.debounce_time:
            return
        if any(data.buttons[self.buttons[direction]] == 1 for direction in ["PS_DIR_UP", "PS_DIR_DOWN", "PS_DIR_LEFT", "PS_DIR_RIGHT"]):
            self.last_button_press_time = rospy.Time.now()

        if data.buttons[self.buttons["PS_DIR_UP"]] == 1:
            # move to start recording
            if self.status==-1:
                self.status=0
            # start teleop
            elif self.status==2:
                self.status=3
            # save and generate next command
            elif self.status==4:
                self.status=5
        elif data.buttons[self.buttons["PS_DIR_DOWN"]] == 1:
            # reset status
            self.has_object=False
            self.object_name = None
            # regenerate command
            if self.status==2:
                self.status=1
            # generate next command without saving
            elif self.status==4:
                self.status=1
        elif data.buttons[self.buttons["PS_DIR_LEFT"]] == 1:
            # stop recording
            if self.status==2:
                self.status=6
            # finish task
            elif self.status==-1:
                self.status=7
        elif data.buttons[self.buttons["PS_DIR_RIGHT"]] == 1:
            rospy.loginfo(f"Change joint state: {self.target_pose_idx}")
            self.change_joint_state(self.target_arm_poses[self.target_pose_idx])
            if self.target_pose_idx == len(self.target_arm_poses)-1:
                self.target_pose_idx=0
            else:
                self.target_pose_idx+=1
    
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

    def initialize_meta_data(self):
        meta_data = dict(
            interface="tidyup_telop",
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

    def speak(self, sentence, language=Voice.kJapanese):
        goal = TalkRequestGoal()
        goal.data.language = language
        goal.data.sentence = sentence
        if (self.talk_request_client.send_goal_and_wait(goal) == GoalStatus.SUCCEEDED):
            rospy.loginfo(f'Successfully spoken: "{sentence}"')
            return True
        else:
            return False

    def process(self):
        rospy.loginfo("==========================")
        rospy.loginfo("UP: start recording") # status: -1->0
        rospy.loginfo("LEFT: QUIT") # status: -1->7
        while not rospy.is_shutdown():
            self.control_mode_pub.publish("manual")
            if self.status==0:
                JST = timezone(timedelta(hours=+9), 'JST')
                task_name = datetime.now(JST).strftime("tidyup-teleop-%y-%m-%d-%H-%M-%S")
                self.start_recording(task_name)
                self.status=1
            elif self.status==1:
                cmd, object_name = self.command_generator.generate_command(has_object=self.has_object, object=self.object_name)
                self.speak(cmd)
                rospy.loginfo("-------------------------")
                rospy.loginfo("command: " + cmd)
                self.status = 2
                rospy.loginfo("-------------------------")
                rospy.loginfo("UP: Start teleop") # status: 2->3
                rospy.loginfo("DOWN: Next Command") # status: 2->1
                rospy.loginfo("LEFT: Finish Tidyup") # status: 2->6
            elif self.status==3:
                self.start_time = time.time()
                self.status=4
                rospy.loginfo("-------------------------")
                rospy.loginfo("UP: Save and Finish Task") # status: 4->5
                rospy.loginfo("DOWN: Next Command without Saving") # status: 4->1
            elif self.status==5:
                end_time = time.time()
                self.record_function_call_time(cmd, self.start_time, end_time, is_successed=True)
                # finish place
                if self.has_object:
                    self.has_object=False
                    self.object_name = None
                # finish pick
                else:
                    self.object_name = object_name
                    self.has_object = True
                self.status=1
            elif self.status==6:
                meta_data_str = json.dumps(self.meta_data)
                self.stop_recording(meta_data_str)
                self.status=-1
                rospy.loginfo("==========================")
                rospy.loginfo("UP: start recording") # status: -1->0
                rospy.loginfo("LEFT: QUIT") # status: -1->7
            elif self.status==7:
                rospy.loginfo("QUIT")
                break
            self.rate.sleep()

def main():
    tidyup_teleop_interface = TidyupTeleopInterface()
    tidyup_teleop_interface.process()

if __name__ == '__main__':
    main()
