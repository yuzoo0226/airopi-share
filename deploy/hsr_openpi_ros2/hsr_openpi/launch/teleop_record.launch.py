#!/usr/bin/env python3
"""Record teleoperation episodes as a ROS 2 bag, on the robot.

Run this **on the HSR** while teleoperating with
`hsr_leader_teleop <https://github.com/Hibikino-Musashi-Home/hsr_leader_teleop>`_
(leader arm + JoyCon, or its keyboard mode). That stack commands the same topics
this deployment uses, so a bag recorded here converts into training data with
``deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py`` unchanged.

    # on the robot
    ros2 launch hsr_openpi teleop_record.launch.py profile:=real \
        output_dir:=/home/administrator/hsr_bags task:="pick up the bottle"

    # per episode
    ros2 service call /hsr_bag_recorder/start_episode \
        hsr_openpi_msgs/srv/StringTrigger "{message: 'pick up the bottle'}"
    ros2 service call /hsr_bag_recorder/stop_episode std_srvs/srv/Trigger
    ros2 service call /hsr_bag_recorder/discard_episode std_srvs/srv/Trigger

Set ``profile:=sim`` to rehearse the whole flow against the Gazebo HSR before
touching hardware; the topic names are the only difference.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_ARGS = (
    ("profile", "real", "Topic set: 'real' (/hsrb/...), 'sim', or 'custom'."),
    ("output_dir", "~/hsr_bags", "Where the bag is written (on this machine)."),
    ("bag_name", "", "Bag directory name; empty means a timestamp."),
    ("task", "teleoperation", "Default task string for episodes."),
    ("storage", "mcap", "rosbag2 storage plugin."),
    ("compression", "", "'' or 'zstd'."),
    ("max_bagfile_size", "0", "Split the bag every N bytes (0 = single file)."),
    ("auto_start", "false", "Open the first episode immediately."),
    ("use_sim_time", "false", "Set true only when recording from the simulator."),
    ("min_episode_seconds", "1.0", "Episodes shorter than this are discarded."),
)


def _launch_setup(context, *args, **kwargs):
    cfg = LaunchConfiguration
    return [
        Node(
            package="hsr_openpi",
            executable="bag_recorder",
            name="hsr_bag_recorder",
            output="screen",
            emulate_tty=True,
            parameters=[
                {
                    "use_sim_time": cfg("use_sim_time"),
                    "profile": cfg("profile"),
                    "output_dir": cfg("output_dir"),
                    "bag_name": cfg("bag_name"),
                    "task": cfg("task"),
                    "storage": cfg("storage"),
                    "compression": cfg("compression"),
                    "max_bagfile_size": cfg("max_bagfile_size"),
                    "auto_start": cfg("auto_start"),
                    "min_episode_seconds": cfg("min_episode_seconds"),
                }
            ],
        )
    ]


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=desc) for n, d, desc in _ARGS]
    return LaunchDescription(declared + [OpaqueFunction(function=_launch_setup)])
