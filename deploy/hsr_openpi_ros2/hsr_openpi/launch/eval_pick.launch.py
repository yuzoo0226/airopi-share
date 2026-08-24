#!/usr/bin/env python3
"""Evaluate a trained policy on the Gazebo pick task.

``hsr_pick_task`` runs in ``drive:=external`` mode: it resets the scene, spawns a
random object at a random pose, publishes the task string and scores the
episode - but it does not move the robot. ``hsr_openpi_node`` drives instead,
from camera images and joint states only.

An episode counts as a success when the object ends up lifted, which requires the
policy to have satisfied the same geometric grasp condition the scripted
demonstrations had to satisfy (finger tips around the object, gripper closing).

    ros2 launch hsr_openpi eval_pick.launch.py num_episodes:=20 \
        bag_path:=/home/hsr/hsr_ros2_ws/_bags/eval_10ep \
        results_path:=/home/hsr/hsr_ros2_ws/_bags/eval_10ep.json
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HEAD_RAW = "/head_rgbd_sensor/rgb/image_rect_color"
HAND_RAW = "/hand_camera/image_raw"
GZ_POSE_TOPIC = "/gz/dynamic_pose"

RECORD_TOPICS = [
    "/clock",
    "/joint_states",
    f"{HEAD_RAW}/compressed",
    f"{HAND_RAW}/compressed",
    "/arm_trajectory_controller/joint_trajectory",
    "/head_trajectory_controller/joint_trajectory",
    "/gripper_controller/joint_trajectory",
    "/omni_base_controller/cmd_vel",
    "/odom",
    "/control_mode",
    "/hsr_pick_task/episode",
    GZ_POSE_TOPIC,
]

_ARGS = (
    ("num_episodes", "20", "Evaluation episodes."),
    ("seed", "1000", "Random seed; keep it away from the collection seed."),
    ("episode_timeout", "25.0", "Seconds the policy gets per episode."),
    ("results_path", "/home/hsr/hsr_ros2_ws/_bags/eval_results.json", "Where the score is written."),
    ("bag_path", "", "Record the evaluation to this bag (empty = no recording)."),
    ("storage", "mcap", "rosbag2 storage plugin."),
    ("world", "default", "Ignition world name."),
    ("use_sim_time", "true", "Use the /clock published by Gazebo."),
    # policy
    ("policy_host", "127.0.0.1", "Policy server host."),
    ("policy_port", "8010", "Policy server port."),
    ("update_freq", "10", "Policy action rate [Hz]."),
    ("adopted_action_chunks", "10", "Chunk elements consumed per inference."),
    ("upsample", "true", "Upsample the action chunk."),
    ("upsample_hz", "50", "Execution rate [Hz]."),
    ("action_smoothing", "ema", "Action smoothing."),
    ("gripper_mode", "continuous", "Gripper mode; the demonstrations use continuous position commands."),
    ("policy_image_order", "bgr", "Channel order handed to the policy."),
    ("instruction", "pick up the object", "Fallback instruction before the first episode message."),
)


def _launch_setup(context, *args, **kwargs):
    cfg = LaunchConfiguration
    world = cfg("world").perform(context)
    bag_path = cfg("bag_path").perform(context)

    actions = [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="gz_pose_bridge",
            output="log",
            arguments=[f"/world/{world}/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V"],
            remappings=[(f"/world/{world}/dynamic_pose/info", GZ_POSE_TOPIC)],
            parameters=[{"use_sim_time": cfg("use_sim_time")}],
        )
    ]

    if bag_path:
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

    scorer = Node(
        package="hsr_openpi",
        executable="pick_task",
        name="hsr_pick_task",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "use_sim_time": cfg("use_sim_time"),
                "drive": "external",
                "num_episodes": cfg("num_episodes"),
                "seed": cfg("seed"),
                "episode_timeout": cfg("episode_timeout"),
                "results_path": cfg("results_path"),
                "world": cfg("world"),
                "object_pose_topic": GZ_POSE_TOPIC,
            }
        ],
    )
    actions.append(scorer)

    policy_launch = os.path.join(
        get_package_share_directory("hsr_openpi"), "launch", "hsr_openpi.launch.py"
    )
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(policy_launch),
            launch_arguments={
                "auto_start": "true",
                "use_sim_time": cfg("use_sim_time"),
                "policy_host": cfg("policy_host"),
                "policy_port": cfg("policy_port"),
                "update_freq": cfg("update_freq"),
                "adopted_action_chunks": cfg("adopted_action_chunks"),
                "upsample": cfg("upsample"),
                "upsample_hz": cfg("upsample_hz"),
                "action_smoothing": cfg("action_smoothing"),
                "gripper_mode": cfg("gripper_mode"),
                "policy_image_order": cfg("policy_image_order"),
                "instruction": cfg("instruction"),
            }.items(),
        )
    )

    actions.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=scorer,
                on_exit=[EmitEvent(event=Shutdown(reason="evaluation finished"))],
            )
        )
    )

    if bag_path:
        os.makedirs(os.path.dirname(bag_path) or ".", exist_ok=True)
        actions.append(
            ExecuteProcess(
                cmd=[
                    "ros2", "bag", "record",
                    "-s", cfg("storage").perform(context),
                    "-o", bag_path,
                    *RECORD_TOPICS,
                ],
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=desc) for n, d, desc in _ARGS]
    return LaunchDescription(declared + [OpaqueFunction(function=_launch_setup)])
