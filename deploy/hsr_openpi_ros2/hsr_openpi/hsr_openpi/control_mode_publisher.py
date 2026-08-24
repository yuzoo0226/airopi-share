#!/usr/bin/env python3
"""Publish ``/control_mode`` so that the openpi node is allowed to act.

On the real robot the DualShock teleop node publishes this topic (``auto`` when
the operator presses the left D-pad). In simulation there is no gamepad, so this
tiny helper lets you flip the mode from the command line::

    ros2 run hsr_openpi control_mode_publisher --ros-args -p mode:=auto
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hsr_openpi.hsr_env import command_qos


class ControlModePublisher(Node):
    def __init__(self) -> None:
        super().__init__("control_mode_publisher")
        self.declare_parameter("mode", "auto")
        self.declare_parameter("topic", "/control_mode")
        self.declare_parameter("rate", 5.0)
        self.mode = str(self.get_parameter("mode").value)
        self.pub = self.create_publisher(String, self.get_parameter("topic").value, command_qos(10))
        period = 1.0 / max(float(self.get_parameter("rate").value), 0.1)
        self.create_timer(period, self._tick)
        self.get_logger().info(f"Publishing control_mode='{self.mode}'")

    def _tick(self) -> None:
        self.pub.publish(String(data=self.mode))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlModePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
