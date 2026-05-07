#!/usr/bin/env python3
import os
import json

import rospy
from actionlib import SimpleActionClient
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortActionGoal
from hsr_data_msgs.srv import StringTrigger, StringTriggerResponse


class NirQuestInterface:
    def __init__(self):
        rospy.init_node('nir_quest_teleop_interface', anonymous=True)

        # parameters
        self.interface = "nir_quest"
        self.hsr_id = rospy.get_param('hsr_id', 'hsrb_000')
        self.location_name = rospy.get_param('location_name', 'location')
        self.instruction = rospy.get_param('~init_instruction', '')
        self.gripper_joint_names = rospy.get_param('gripper_joint_names', ['hand_motor_joint'])
        self.hash_value = os.getenv('GIT_HASH')
        self.branch_name = os.getenv('GIT_BRANCH')
        self.task_name = self.get_task_name()

        self.joint_state = None
        self.is_recording = None
        self.gripper_state = 0
        self.teleop_controller = None
        self.current_mode = "manual"

        self.last_button_press_time = rospy.Time.now()
        self.debounce_time = rospy.Duration(1.0)
        self.rate = rospy.Rate(10)

        # publisher
        self.control_mode_pub = rospy.Publisher('/control_mode', String, queue_size=10)
        self.gripper_open_pub = rospy.Publisher("/hsrb/gripper_controller/command", JointTrajectory, queue_size=10)
        self.gripper_close_client = SimpleActionClient("/hsrb/gripper_controller/grasp", GripperApplyEffortAction)
        self.gripper_close_client.wait_for_server()

        # subscriber
        rospy.Subscriber('/nir/quest/left/joy', Joy, self.left_joy_callback)
        rospy.Subscriber('/nir/quest/right/joy', Joy, self.right_joy_callback)
        rospy.Subscriber("/hsrb/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber('/rosbag_manager/recording', Bool, self.rosbag_status_callback)
        rospy.Subscriber("/hsrb/gripper_controller/command", JointTrajectory, self.gripper_open_callback)
        rospy.Subscriber("/hsrb/gripper_controller/grasp/goal", GripperApplyEffortActionGoal, self.gripper_close_callback)

        # service
        rospy.Service('/update_instruction', StringTrigger, self.update_instruction_srv)
        self.start_recording = rospy.ServiceProxy('/rosbag_manager/start_recording', StringTrigger)
        self.stop_recording = rospy.ServiceProxy('/rosbag_manager/stop_recording', StringTrigger)
        self.delete_recording = rospy.ServiceProxy('/rosbag_manager/delete_recording', Trigger)

    def joint_state_callback(self, data):
        self.joint_state = data

    def rosbag_status_callback(self, data):
        self.is_recording = data.data

    def gripper_open_callback(self, _):
        self.gripper_state = 1

    def gripper_close_callback(self, _):
        self.gripper_state = 0

    def left_joy_callback(self, data):
        if data.buttons[3]:
            self.teleop_controller = "left"
        else:
            self.joy_callback(data)
    
    def right_joy_callback(self, data):
        if data.buttons[3]:
            self.teleop_controller = "right"
        else:
            self.joy_callback(data)

    def joy_callback(self, data):
        if rospy.Time.now() - self.last_button_press_time < self.debounce_time:
            return
        if self.is_recording is None:
            return 
        if self.joint_state is None:
            return
        # Toggle rosbag record start/stop
        if data.buttons[0] == 1:
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
        elif data.buttons[1] == 1:
            self.last_button_press_time = rospy.Time.now()
            self.delete_recording()
        # Change control mode
        elif data.axes[3] > 0.8:
            self.last_button_press_time = rospy.Time.now()
            if self.current_mode == "manual":
                self.current_mode = "auto"
            else:
                self.current_mode = "manual"
            rospy.loginfo(f"Control mode changed to: {self.current_mode}")
        # Toggle gripper open/close
        elif data.axes[4] > 0.8:
            self.last_button_press_time = rospy.Time.now()
            if self.gripper_state == 1: # open -> close
                goal = GripperApplyEffortActionGoal()
                goal.goal.effort = -0.018
                self.gripper_close_client.send_goal(goal.goal)
            else: # close -> open
                traj = JointTrajectory()
                traj.joint_names = self.gripper_joint_names
                p = JointTrajectoryPoint()
                p.positions = [1.239183768915974]
                p.velocities = []
                p.time_from_start = rospy.Duration(1)
                traj.points = [p]
                self.gripper_open_pub.publish(traj)
            self.frames_since_last_pressed = 0
    
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

    def process(self):
        while not rospy.is_shutdown():
            self.control_mode_pub.publish(self.current_mode)
            self.rate.sleep()

        if self.is_recording:
            self.delete_recording()

def main():
    nir_quest_interface = NirQuestInterface()
    nir_quest_interface.process()

if __name__ == '__main__':
    main()
