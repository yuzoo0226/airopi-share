#!/usr/bin/env python3
import subprocess

import numpy as np

import rospy
from actionlib import SimpleActionClient
from sensor_msgs.msg import Joy, JointState
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortActionGoal


class SpaceMouseTeleop:
    """
    Basic teleop interface with SpaceMouse mainly for 4-dof top-down grasping
    """
    def __init__(self):
        rospy.init_node("spacemouse_teleop")
        
        # parameters
        self.arm_joint_names = rospy.get_param('arm_joint_names', [])
        self.gripper_joint_names = rospy.get_param('gripper_joint_names', [])

        # publishers
        self.arm_pub = rospy.Publisher("/hsrb/arm_trajectory_controller/command", JointTrajectory, queue_size=10)
        self.gripper_open_pub = rospy.Publisher("/hsrb/gripper_controller/command", JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=10)
        self.gripper_close_client = SimpleActionClient("/hsrb/gripper_controller/grasp", GripperApplyEffortAction)
        self.gripper_close_client.wait_for_server()

        # subscribers
        rospy.Subscriber("/spacenav/joy", Joy, self.spacenav_callback)
        rospy.Subscriber("/hsrb/gripper_controller/command", JointTrajectory, self.gripper_open_callback)
        rospy.Subscriber("/hsrb/gripper_controller/grasp/goal", GripperApplyEffortActionGoal, self.gripper_close_callback)
        rospy.Subscriber("/hsrb/joint_states", JointState, self.joint_state_callback)

        self.gripper_state = 0
        self.joint_state = None
        self.buttons = None
        self.button_history = []
        self.frames_since_last_pressed = 100

    def gripper_open_callback(self, _):
        self.gripper_state = 1

    def gripper_close_callback(self, _):
        self.gripper_state = 0

    def joint_state_callback(self, data):
        self.joint_state = data

    def spacenav_callback(self, data):
        self.button_history.append(data.buttons)
        self.button_history = self.button_history[-10:]

        if self.buttons is None:
            self.buttons = data.buttons[0]
            gripper_pressed = False
            base_pressed = False
        else:
            # smooth values
            curr_pressed = np.asarray(self.button_history[-2:]).max(0)
            already_pressed = np.asarray(self.button_history[-3:-1]).max(0)
            gripper_pressed = curr_pressed[0] and not already_pressed[0]
            base_pressed = curr_pressed[1]

        self.frames_since_last_pressed += 1

        if np.all(np.asarray(data.axes) == 0) and not gripper_pressed:
            # no command
            return
            
        pos_offset = np.asarray(data.axes[:3]) / 0.68
        rot_offset = np.asarray(data.axes[3:]) / 0.68
        # move base
        msg = Twist()
        base_vel = 0.3 if base_pressed else 0.1
        msg.linear.x = pos_offset[0] * base_vel
        msg.linear.y = pos_offset[1] * base_vel
        if base_pressed:
            msg.angular.z  = rot_offset[2]
        self.base_pub.publish(msg)

        # arm
        current_arm_lift_joint_pos = self.joint_state.position[self.joint_state.name.index('arm_lift_joint')]
        current_wrist_roll_joint_pos = self.joint_state.position[self.joint_state.name.index('wrist_roll_joint')]
        if not base_pressed:
            arm_lift_joint_pos = current_arm_lift_joint_pos + pos_offset[2] * 0.05
            wrist_roll_joint_pos = current_wrist_roll_joint_pos - 0.5 * float(abs(rot_offset[2]) > 0.5) * np.sign(rot_offset[2])
            traj = JointTrajectory()
            traj.joint_names = self.arm_joint_names
            p = JointTrajectoryPoint()
            p.positions = [
                self.joint_state.position[self.joint_state.name.index('arm_flex_joint')],
                arm_lift_joint_pos,
                self.joint_state.position[self.joint_state.name.index('arm_roll_joint')],
                self.joint_state.position[self.joint_state.name.index('wrist_flex_joint')],
                wrist_roll_joint_pos
                ]
            p.velocities = []
            p.time_from_start = rospy.Duration(0.1)
            traj.points = [p]
            self.arm_pub.publish(traj)

        # gripper
        if gripper_pressed and self.frames_since_last_pressed > 30:
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

def main():
    spacemouse_teleop = SpaceMouseTeleop()
    rospy.spin()

if __name__ == "__main__":
    main()
