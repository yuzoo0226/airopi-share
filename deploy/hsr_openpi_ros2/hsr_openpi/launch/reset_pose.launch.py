#!/usr/bin/env python3
"""Send the HSR to a known start pose."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    profile = LaunchConfiguration("robot_profile").perform(context)
    share = get_package_share_directory("hsr_openpi")
    topics_yaml = os.path.join(share, "config", f"{profile}_topics.yaml")
    return [
        Node(
            package="hsr_openpi",
            executable="reset_pose",
            name="reset_pose",
            output="screen",
            emulate_tty=True,
            parameters=[
                topics_yaml,
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "pose_file": LaunchConfiguration("pose_file"),
                    "duration": LaunchConfiguration("duration"),
                },
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_profile", default_value="sim"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("pose_file", default_value=""),
            DeclareLaunchArgument("duration", default_value="5.0"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
