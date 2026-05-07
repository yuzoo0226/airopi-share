#!/usr/bin/env python3
import os
import json
import subprocess
from datetime import datetime, timezone, timedelta

import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse
from hsr_data_msgs.srv import StringTrigger, StringTriggerResponse

class RosbagController:
    def __init__(self):
        rospy.init_node('rosbag_controller', anonymous=True)

        self.record_topics = rospy.get_param('record_topics', ['/tf'])
        self.record_base_dir = rospy.get_param('record_base_dir', '/root/datasets/rosbags')
        self.hsr_id = rospy.get_param('hsr_id', 'hsrb_000')
        self.location_name = rospy.get_param('location_name', 'location')
        self.hash_value = os.getenv('GIT_HASH')
        self.branch_name = os.getenv('GIT_BRANCH')

        self.rosbag_process = None
        self.is_recording = False
        self.recent_bagfile_name = None

        self.rosbag_manger_mode_pub = rospy.Publisher('/rosbag_manager/recording', Bool, queue_size=10)
        rospy.Service('/rosbag_manager/start_recording', StringTrigger, self.start_recording_srv)
        rospy.Service('/rosbag_manager/stop_recording', StringTrigger, self.stop_recording_srv)
        rospy.Service('/rosbag_manager/delete_recording', Trigger, self.delete_recording_srv)
        rospy.Service('/rosbag_manager/toggle_recording', StringTrigger, self.toggle_recording_srv)

        self.rate = rospy.Rate(1)

    def start_recording(self, message):
        self.task_name = message
        if not self.is_recording:
            JST = timezone(timedelta(hours=+9), 'JST')
            record_filename = datetime.now(JST).strftime(
                os.path.join(self.record_base_dir, self.task_name,
                             "demo-%y-%m-%d-%H-%M-%S.bag")
                            )
            if not os.path.exists(os.path.dirname(record_filename)):
                os.makedirs(os.path.dirname(record_filename))
            command = ['rosbag', 'record', '-O', record_filename]
            command.extend(self.record_topics)
            self.rosbag_process = subprocess.Popen(command)
            self.is_recording = True
            self.recent_bagfile_name = record_filename
            rospy.loginfo("Rosbag recording started.")
            return True
        else:
            rospy.logwarn("Rosbag is already recording.")
            return False

    def stop_recording(self, message):
        if self.is_recording:
            self.save_meta_file(message)
            self.rosbag_process.send_signal(subprocess.signal.SIGINT)
            self.rosbag_process.wait() 
            self.rosbag_process = None
            self.is_recording = False

            rospy.loginfo("Rosbag recording stopped.")
            return True
        else:
            rospy.logwarn("Rosbag is not recording.")
            return False
    
    def toggle_recording(self, message):
        if not self.is_recording:
            res = self.start_recording(message)
        else:
            res = self.stop_recording(message)
        return res

    def delete_recording(self):
        if self.recent_bagfile_name and os.path.exists(self.recent_bagfile_name):
            os.remove(self.recent_bagfile_name)
            rospy.loginfo(f"Recent rosbag file {self.recent_bagfile_name} has been removed.")
            self.delete_meta_data()
            self.recent_bagfile_name = None
            return True
        else:
            rospy.logwarn("No recent rosbag file to remove or file does not exist.")
            return False

    def start_recording_srv(self, req):
        res = self.start_recording(req.message)
        return StringTriggerResponse(success=res)

    def stop_recording_srv(self, req):
        res = self.stop_recording(req.message)
        return StringTriggerResponse(success=res)

    def toggle_recording_srv(self, req):
        res = self.toggle_recording(req.message)
        return StringTriggerResponse(success=res)

    def delete_recording_srv(self, req):
        if self.is_recording:
            res = self.stop_recording(message=None)
            if res:
                res = self.delete_recording()
        else:
            res = self.delete_recording()
        return TriggerResponse(success=res)

    def init_meta_data(self):
        meta_data =dict(
            bag_path=self.task_name,
            hsr_id=self.hsr_id,
            location_name=self.location_name,
            interface="",
            instructions=[],
            segments=[],
            git_hash=self.hash_value,
            git_branch=self.branch_name,
            interface_git_hash="",
            interface_git_branch="",
        )
        return meta_data

    def save_meta_file(self, message=None):
        if message is None:
            return 
        data = self.init_meta_data()
        message_data = json.loads(message)
        for msg_key, msg_value in message_data.items():
            if msg_key in data.keys():
                data[msg_key] = msg_value
        if len(data["instructions"])==0 or \
            len(data["segments"])==0 or \
            data["interface"]=="":
            return

        meta_data_file = os.path.join(self.record_base_dir, "meta.json")
        if os.path.exists(meta_data_file):
            with open(meta_data_file, 'r') as f:
                meta_data = json.load(f)
            is_duplicate = False
            for entry in meta_data:
                if all(entry.get(key)==data.get(key) for key in ["bag_path"]):
                    is_duplicate = True
                    break
            if not is_duplicate:
                meta_data.append(data)
        else:
            meta_data = [data]
        with open(meta_data_file, 'w') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

    def delete_meta_data(self):
        meta_data_file = os.path.join(self.record_base_dir, "meta.json")
        dir_files = os.listdir(os.path.join(os.path.dirname(self.recent_bagfile_name)))
        rosbag_files = [file for file in dir_files if file.endswith('.bag')]
        if len(rosbag_files)==0:
            with open(meta_data_file, 'r') as f:
                meta_data = json.load(f)
            meta_data = [entry for entry in meta_data if entry.get("bag_path") != self.task_name]
            if len(meta_data)>0:
                with open(meta_data_file, 'w') as f:
                    json.dump(meta_data, f, ensure_ascii=False, indent=2)
            else:
                os.remove(meta_data_file)
            rospy.loginfo("Delete recent meta data.")
        
    def process(self):
        while not rospy.is_shutdown():
            self.rosbag_manger_mode_pub.publish(self.is_recording)
            self.rate.sleep()

def main():
    rosbag_controller = RosbagController()
    rosbag_controller.process()

if __name__ == '__main__':
    main()
