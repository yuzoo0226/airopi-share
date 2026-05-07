#!/usr/bin/env python3
import os
import json
import struct
import socket

import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from actionlib import SimpleActionClient
from tf import TransformListener, TransformBroadcaster
from geometry_msgs.msg import Twist, Point, PoseStamped, TransformStamped, WrenchStamped
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from hsr_data_msgs.srv import StringTrigger, StringTriggerResponse


class QuestInterface:
    def __init__(self):
        rospy.init_node('quest_interface', anonymous=True)

        # parameters
        self.interface = rospy.get_param('~quest_interface', 'quest2')
        self.hsr_id = rospy.get_param('hsr_id', 'hsrb_000')
        self.location_name = rospy.get_param('location_name', 'location')
        self.instruction = rospy.get_param('~init_instruction', '')
        self.hash_value = os.getenv('GIT_HASH')
        self.branch_name = os.getenv('GIT_BRANCH')
        self.task_name = self.get_task_name()
        self.arm_joint_names = rospy.get_param('arm_joint_names', [])
        self.head_joint_names = rospy.get_param('head_joint_names', [])
        self.gripper_joint_names = rospy.get_param('gripper_joint_names', [])
        self.target_arm_poses = [
            rospy.get_param('top_down_joint_poses', {}),
            rospy.get_param('pick_up_joint_poses', {}),
            rospy.get_param('home_joint_poses', {}),
        ]

        self.last_data = None
        self.pre_ee_control_mode = False
        self.controller_origin = None
        self.hand_origin = None
        self.joint_state = None
        self.wrist_data = None
        self.gripper_state = 0
        self.is_recording = None
        self.current_mode = "manual"
        self.target_pose_idx = 0
        self.debounce_time = rospy.Duration(1.0)
        self.last_trigger_press_time = rospy.Time.now()
        self.host = '0.0.0.0'
        self.port = 8000
        self.header_size = 4
        self.rate = rospy.Rate(10)

        # publisher
        self.control_mode_pub = rospy.Publisher('/control_mode', String, queue_size=10)
        self.ee_pub = rospy.Publisher('/hsrb/pseudo_endeffector_position_controller/command_position', PoseStamped, queue_size=10)
        self.ee_with_base_pub = rospy.Publisher('/hsrb/pseudo_endeffector_position_controller/command_position_with_base', PoseStamped, queue_size=10)
        self.base_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)
        self.arm_pub = rospy.Publisher("/hsrb/arm_trajectory_controller/command", JointTrajectory, queue_size=10)
        self.head_pub = rospy.Publisher("/hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=10)
        self.gripper_open_pub = rospy.Publisher("/hsrb/gripper_controller/command", JointTrajectory, queue_size=10)
        self.gripper_close_client = SimpleActionClient("/hsrb/gripper_controller/grasp", GripperApplyEffortAction)
        self.gripper_close_client.wait_for_server()

        # subscriber
        rospy.Subscriber('/rosbag_manager/recording', Bool, self.rosbag_status_callback)
        rospy.Subscriber("/hsrb/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber('/hsrb/wrist_wrench/compensated', WrenchStamped, self.wrist_callback)
        rospy.Subscriber("/hsrb/gripper_controller/command", JointTrajectory, self.gripper_open_callback)
        rospy.Subscriber("/hsrb/gripper_controller/grasp/goal", GripperApplyEffortActionGoal, self.gripper_close_callback)

        # service
        rospy.Service('/update_instruction', StringTrigger, self.update_instruction_srv)
        self.start_recording = rospy.ServiceProxy('/rosbag_manager/start_recording', StringTrigger)
        self.stop_recording = rospy.ServiceProxy('/rosbag_manager/stop_recording', StringTrigger)
        self.delete_recording = rospy.ServiceProxy('/rosbag_manager/delete_recording', Trigger)

        # transform broadcaster
        self.tf_broadcaster = TransformBroadcaster()
        # tranform listener
        self.tf_listener = TransformListener()

    def gripper_open_callback(self, data):
        self.gripper_state = 1

    def gripper_close_callback(self, data):
        self.gripper_state = 0

    def joint_state_callback(self, data):
        self.joint_state = data

    def wrist_callback(self, data):
        self.wrist_data = data

    def rosbag_status_callback(self, data):
        self.is_recording = data.data

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

    def receive_message(self, conn, msg_size):
        data = b''
        while len(data) < msg_size:
            chunk = conn.recv(msg_size - len(data))
            if not chunk:
                break
            data += chunk
        return data.decode()

    def send_message(self, socket, message):
        message_encoded = message.encode()
        # Create the header with 2 bytes representing the length of the message
        header = len(message_encoded).to_bytes(2, byteorder='big')
        # Send the header followed by the message
        socket.send(header + message_encoded)
    
    def transform_controller(self, pose):
        trans = np.eye(4)
        trans[:3, :3] = R.from_euler('x', 0.5*np.pi).as_matrix()
        mat = trans.dot(pose)
        trans[:3, :3] = R.from_euler('yz', [np.pi, -0.5*np.pi]).as_matrix()
        mat = mat.dot(trans)
        return mat

    def broadcast_tf(self, pose, parent, child):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent
        t.child_frame_id = child

        mat = np.asarray(pose)
        t.transform.translation.x = mat[0,3]
        t.transform.translation.y = mat[1,3]
        t.transform.translation.z = mat[2,3]

        q = R.from_matrix(mat[:3, :3]).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(
            (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z),
            (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w),
            rospy.Time.now(),
            t.child_frame_id,
            t.header.frame_id
        )

    def tf2mat(self, pos, rot):
        x = np.eye(4)
        x[:3, 3] = pos
        x[:3, :3] = R.from_quat(rot).as_matrix()
        return x
    
    def get_tf_as_matrix(self, to_frame, from_frame):
        translation, rotation = self.tf_listener.lookupTransform(to_frame, from_frame, rospy.Time(0))
        mat = self.tf2mat(translation, rotation)
        return mat

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

    def toggle_gripper_state(self):
        if self.gripper_state==1: # open -> close
            goal = GripperApplyEffortActionGoal()
            goal.goal.effort = -0.0018
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

    def update_trigger_action(self, data):
        if rospy.Time.now() - self.last_trigger_press_time < self.debounce_time:
            return
        # Toggle rosbag record start/stop
        if data["L"]["inputs"]["ulButtonPressed"]==128: # X
            self.last_trigger_press_time = rospy.Time.now()
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
        # Delete recent rosbag file
        elif data["L"]["inputs"]["ulButtonPressed"]==2: # Y
            self.last_button_press_time = rospy.Time.now()
            self.delete_recording()
        # Change Joint State
        elif data["R"]["inputs"]["ulButtonPressed"]==128: # A
            if self.joint_state is None:
                return
            self.last_trigger_press_time = rospy.Time.now()
            if self.current_mode == "manual" and not self.is_recording:
                rospy.loginfo(f"Change joint state: {self.target_pose_idx}")
                self.change_joint_state(self.target_arm_poses[self.target_pose_idx])
                if self.target_pose_idx == len(self.target_arm_poses)-1:
                    self.target_pose_idx=0
                else:
                    self.target_pose_idx+=1
        # Change Control Mode
        elif data["R"]["inputs"]["ulButtonPressed"]==2: # B
            self.last_button_press_time = rospy.Time.now()
            if self.current_mode == "manual":
                self.current_mode = "auto"
            else:
                self.current_mode = "manual"
            rospy.loginfo(f"Control mode changed to: {self.current_mode}")
        # Gripper
        elif data["R"]["inputs"]["trackpad_pressed"]==1:
            self.last_trigger_press_time = rospy.Time.now()
            if self.current_mode=="manual":
                self.toggle_gripper_state()
    
    def dampen(self, rel_pose, coef):
        rel_pose[:3, 3] *= coef
        a = np.asarray(R.from_matrix(rel_pose[:3, :3]).as_quat())
        b = np.asarray([0,0,0,1], dtype=np.float32)
        new_quat = R.from_quat((1-coef)*b+coef*a).as_quat()
        rel_pose[:3, :3] = R.from_quat(new_quat).as_matrix()
        return rel_pose

    def move_base(self, x, y, z, faster=False):
        speed_linear = 0.38 if faster else 0.1
        speed_angular = 1.5 if faster else 0.5
        msg = Twist()
        msg.linear.x = x * speed_linear
        msg.linear.y = y * speed_linear
        msg.angular.z = z * speed_angular
        self.base_pub.publish(msg)

    def ee_control(self, pose, frame_id, with_base=True):
        ee = PoseStamped()
        ee.header.frame_id = frame_id
        ee.pose.position = Point(
            x = pose[0,3],
            y = pose[1,3],
            z = pose[2,3],
        )
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        ee.pose.orientation.x = quat[0]
        ee.pose.orientation.y = quat[1]
        ee.pose.orientation.z = quat[2]
        ee.pose.orientation.w = quat[3]
        if with_base:
            self.ee_with_base_pub.publish(ee)
        else:
            self.ee_pub.publish(ee)

    def process(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        while not rospy.is_shutdown():
            conn, addr =server_socket.accept()
            rospy.loginfo("Connection from: " + str(addr))
            while not rospy.is_shutdown():
                header = conn.recv(self.header_size)
                if not header:
                    rospy.logwarn("Connection closed.")
                msg_size = struct.unpack('!I', header)[0]
                data = self.receive_message(conn, msg_size)
                # sending data
                if self.wrist_data is not None:
                    wrist_force = self.wrist_data.wrench.force
                    wrist_force = np.linalg.norm([wrist_force.x, wrist_force.y, wrist_force.z])
                    self.send_message(conn, f"{wrist_force}")
                if not data:
                    rospy.loginfo("No data received.")
                    continue
                data = json.loads(data)
                for device_id in ["controller_1", "controller_2"]:
                    pose  = data[device_id]["pose"]
                    pose.append([0,0,0,1])
                    updated_pose = self.transform_controller(pose)
                    data[device_id]['pose'] = updated_pose
                    self.broadcast_tf(updated_pose, "odom", device_id)
                data = {"L": data['controller_1'], "R": data['controller_2']}
                if self.last_data is None:
                    self.last_data = data
                    continue
                self.update_trigger_action(data)

                if self.current_mode == "manual":
                    pre_ee_control_mode = self.last_data["R"]["inputs"]["grip_button"]==1
                    speed_trigger = data["L"]["inputs"]["trigger"] > 0.5
                    with_base = data["R"]["inputs"]["trigger"] > 0.5
                    # move base
                    if data["L"]["inputs"]["grip_button"]==1:
                        x = data["L"]["inputs"]["trackpad_y"]
                        y = -data["L"]["inputs"]["trackpad_x"]
                        z = -data["R"]["inputs"]["trackpad_x"]
                        self.move_base(x,y,z,faster=speed_trigger)
                    ## end effector control
                    if data["R"]["inputs"]["grip_button"]==1:
                        if not pre_ee_control_mode:
                            self.controller_origin = np.asarray(data["L"]["pose"])
                            self.hand_origin = self.get_tf_as_matrix("odom", "hand_palm_link")
                            self.broadcast_tf(self.hand_origin, 'odom', 'hand_origin')
                        if self.hand_origin is not None:
                            rel_mat = np.linalg.inv(self.controller_origin).dot(np.asarray(data["L"]["pose"]))
                            coef = 0.25 if speed_trigger else 1
                            rel_mat = self.dampen(rel_mat, coef)
                            ee_target_mat = self.hand_origin.dot(rel_mat)
                            self.ee_control(ee_target_mat, 'odom', with_base=with_base)
                            self.broadcast_tf(ee_target_mat, 'odom', 'ee_target')

                self.control_mode_pub.publish(self.current_mode)
                self.last_data = data
                self.rate.sleep()

        if self.is_recording:
            self.delete_recording()

def main():
    quest_interface = QuestInterface()
    quest_interface.process()

if __name__ == '__main__':
    main()
