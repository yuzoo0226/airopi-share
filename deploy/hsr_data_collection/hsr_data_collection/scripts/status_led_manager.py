#!/usr/bin/env python3
import rospy
from actionlib import SimpleActionClient
from std_msgs.msg import String, ColorRGBA, Bool
from std_srvs.srv import Empty
from tmc_msgs.msg import GradationalColorGoal, GradationalColorAction
from tmc_msgs.srv import SetColor


class StatusLEDManager:
    def __init__(self):
        rospy.init_node('status_led_manager', anonymous=True)

        # parameters
        self.is_recording = None
        self.led_color = None
        self.current_mode = None
        self.auto_mode_color = ColorRGBA(r=0,g=1,b=1,a=0)
        self.manual_mode_color = ColorRGBA(r=1,g=0,b=1,a=0)
        self.rate = rospy.Rate(10)

        # publisher
        self.led_service = rospy.ServiceProxy('/hsrb/status_led_node/set_color', SetColor)

        # subscriber
        rospy.Subscriber('/rosbag_manager/recording', Bool, self.rosbag_status_callback)
        rospy.Subscriber('/hsrb/command_status_led_rgb', ColorRGBA, self.status_led_callback)
        rospy.Subscriber('/control_mode', String, self.control_mode_callback)

        # service
        self.status_led_auto_mode = rospy.ServiceProxy("/hsrb/status_led_node/activate_auto_mode", Empty)
        self.led_client = SimpleActionClient("/hsrb/gradational_color", GradationalColorAction)

    def rosbag_status_callback(self, data):
        self.is_recording = data.data
    
    def status_led_callback(self, data):
        self.led_color = data

    def control_mode_callback(self, data):
        self.current_mode = data.data

    def start_pulsing(self, duration=1.0):
        goal = GradationalColorGoal()
        blink_color = ColorRGBA(r=0,g=1,b=0,a=0)
        goal.base_colors = [blink_color]
        goal.blink_interval = rospy.Duration(duration)
        goal.blink_count = int(1e6)
        goal.do_not_overwrite = False
        self.led_client.send_goal(goal)

    def stop_pulsing(self):
        self.led_client.cancel_goal()

    def process(self):
        is_pulsing = False
        base_color = self.manual_mode_color
        while not rospy.is_shutdown():
            if self.is_recording is None:
                continue
            if self.led_color is None:
                continue
            if self.current_mode is None:
                continue

            if not is_pulsing and self.is_recording:
                self.start_pulsing()
                is_pulsing = True
            elif is_pulsing and not self.is_recording:
                self.stop_pulsing()
                is_pulsing = False

            if not is_pulsing:
                if self.current_mode == "manual" and self.led_color!=self.manual_mode_color:
                    self.led_service(self.manual_mode_color, False)
                    base_color = self.manual_mode_color
                elif self.current_mode == "auto" and self.led_color!=self.auto_mode_color:
                    self.led_service(self.auto_mode_color, False)
                    base_color = self.auto_mode_color
            elif self.is_recording:
                if self.current_mode == "manual" and base_color!=self.manual_mode_color:
                    self.stop_pulsing()
                    self.led_service(self.manual_mode_color, False)
                    base_color = self.manual_mode_color
                    self.start_pulsing()
                elif self.current_mode == "auto" and base_color!=self.auto_mode_color:
                    self.stop_pulsing()
                    self.led_service(self.auto_mode_color, False)
                    base_color = self.auto_mode_color
                    self.start_pulsing()
            self.rate.sleep()

        self.status_led_auto_mode()

def main():
    status_led_manager = StatusLEDManager()
    status_led_manager.process()

if __name__ == '__main__':
    main()
