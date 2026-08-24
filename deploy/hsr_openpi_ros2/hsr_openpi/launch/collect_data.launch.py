#!/usr/bin/env python3
"""Record a ROS 2 bag of the HSR moving randomly in Gazebo.

Starts three things:

1. ``image_transport republish`` for both cameras, so the bag carries
   ``.../compressed`` topics with the same names the real robot records
   (Ignition only publishes raw images, and raw 640x480 frames make the bag
   roughly 30x bigger).
2. ``hsr_random_motion``, which drives the arm / head / gripper / base with a
   clamped Ornstein-Uhlenbeck random walk and publishes ``/control_mode``.
3. ``ros2 bag record`` on the topic set that
   ``deploy/hsr_openpi_ros2/tools/rosbag2_to_lerobot.py`` consumes.

Example
-------
    ros2 launch hsr_openpi collect_data.launch.py \
        bag_path:=/home/hsr/hsr_ros2_ws/_bags/random_01 \
        num_episodes:=5 episode_duration:=30.0 task:="move the arm around randomly"
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HEAD_RAW = "/head_rgbd_sensor/rgb/image_rect_color"
HAND_RAW = "/hand_camera/image_raw"

RECORD_TOPICS = [
    "/clock",
    "/joint_states",
    f"{HEAD_RAW}/compressed",
    "/head_rgbd_sensor/rgb/camera_info",
    f"{HAND_RAW}/compressed",
    "/hand_camera/camera_info",
    "/arm_trajectory_controller/joint_trajectory",
    "/head_trajectory_controller/joint_trajectory",
    "/gripper_controller/joint_trajectory",
    "/omni_base_controller/cmd_vel",
    "/odom",
    "/tf",
    "/tf_static",
    "/control_mode",
    "/hsr_random_motion/episode",
]

_ARGS = (
    ("bag_path", "/home/hsr/hsr_ros2_ws/_bags/random", "Output bag directory (must not exist)."),
    ("storage", "mcap", "rosbag2 storage plugin: 'mcap' or 'sqlite3'."),
    ("num_episodes", "5", "Number of episodes to record (0 = until stopped)."),
    ("episode_duration", "30.0", "Seconds of random motion per episode."),
    ("reset_duration", "6.0", "Seconds spent returning to the start pose between episodes."),
    ("update_freq", "10.0", "Command rate [Hz]; also the rate the dataset is resampled to."),
    ("seed", "0", "Random seed."),
    ("move_base", "true", "Also drive the omni base."),
    ("task", "move the arm around randomly", "Task string stored as the LeRobot task / prompt."),
    ("record", "true", "Set to false to drive the robot without recording."),
    ("use_sim_time", "true", "Use the /clock published by Gazebo."),
)


def _launch_setup(context, *args, **kwargs):
    cfg = LaunchConfiguration
    bag_path = cfg("bag_path").perform(context)
    storage = cfg("storage").perform(context)
    do_record = cfg("record").perform(context).lower() in ("1", "true", "yes")

    actions = []

    for raw_topic in (HEAD_RAW, HAND_RAW):
        name = raw_topic.strip("/").replace("/", "_")
        actions.append(
            Node(
                package="image_transport",
                executable="republish",
                name=f"republish_{name}",
                output="log",
                arguments=["raw", "compressed"],
                remappings=[("in", raw_topic), ("out/compressed", f"{raw_topic}/compressed")],
                parameters=[{"use_sim_time": cfg("use_sim_time")}],
            )
        )

    motion_node = Node(
            package="hsr_openpi",
            executable="random_motion",
            name="hsr_random_motion",
            output="screen",
            emulate_tty=True,
            parameters=[
                {
                    "use_sim_time": cfg("use_sim_time"),
                    "num_episodes": cfg("num_episodes"),
                    "episode_duration": cfg("episode_duration"),
                    "reset_duration": cfg("reset_duration"),
                    "update_freq": cfg("update_freq"),
                    "seed": cfg("seed"),
                    "move_base": cfg("move_base"),
                    "task": cfg("task"),
                }
            ],
    )
    actions.append(motion_node)
    # random_motion exits once num_episodes are done; tearing the launch down
    # from there sends SIGINT to `ros2 bag record`, which is what makes it write
    # metadata.yaml. Killing the recorder instead leaves a bag that neither
    # `ros2 bag info` nor the converter can open without a manual reindex.
    actions.append(
        RegisterEventHandler(
            OnProcessExit(target_action=motion_node, on_exit=[EmitEvent(event=Shutdown(reason="collection finished"))])
        )
    )

    if do_record:
        os.makedirs(os.path.dirname(bag_path) or ".", exist_ok=True)
        actions.append(
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-s", storage, "-o", bag_path, *RECORD_TOPICS],
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=desc) for n, d, desc in _ARGS]
    return LaunchDescription(declared + [OpaqueFunction(function=_launch_setup)])
