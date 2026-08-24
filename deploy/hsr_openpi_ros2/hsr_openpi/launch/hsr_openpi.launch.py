#!/usr/bin/env python3
"""Launch the ROS 2 openpi inference node for the HSR.

Examples
--------
Simulation, policy served by ``docker compose up openpi-server``::

    ros2 launch hsr_openpi hsr_openpi.launch.py \
        instruction:="Grasp the bottle." auto_start:=true

Real robot::

    ros2 launch hsr_openpi hsr_openpi.launch.py robot_profile:=real \
        policy_host:=192.168.1.10
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_ARGS = (
    ("robot_profile", "sim", "Topic layout: 'sim' (Ignition Gazebo) or 'real' (hsrb bringup)."),
    ("use_sim_time", "true", "Use the /clock published by Gazebo."),
    ("policy_backend", "websocket", "'websocket' (policy server) or 'local' (openpi in-process)."),
    ("policy_host", "127.0.0.1", "Policy server host."),
    ("policy_port", "8010", "Policy server port."),
    ("policy_connect_timeout", "-1.0", "Seconds to wait for the policy server (-1 = forever)."),
    ("config_name", "", "Registered openpi TrainConfig name (policy_backend:=local)."),
    ("config_yaml", "", "Experiment YAML path (policy_backend:=local)."),
    ("checkpoint_dir", "", "Checkpoint step directory (policy_backend:=local)."),
    ("instruction", "Grasp the bottle.", "Initial language instruction."),
    ("update_freq", "10", "Policy action rate [Hz]."),
    ("adopted_action_chunks", "10", "Number of chunk elements consumed per inference."),
    ("upsample", "true", "Upsample the action chunk to a higher execution rate."),
    ("upsample_hz", "100", "Execution rate when upsample is enabled [Hz]."),
    ("upsample_method", "spline", "'spline' or 'linear'."),
    ("action_smoothing", "ema", "'none', 'ema' or 'moving_average'."),
    ("ema_alpha", "0.2", "EMA coefficient."),
    ("ma_window", "5", "Moving average window."),
    ("smooth_gripper", "false", "Also smooth the gripper dimension."),
    ("smooth_base", "false", "Also smooth the base dimensions."),
    ("gripper_mode", "hybrid", "'continuous', 'discrete' or 'hybrid'."),
    ("policy_image_order", "bgr", "Channel order handed to the policy: 'bgr' or 'rgb'. See docs/ros2_deploy.md 6.1."),
    ("require_control_mode", "true", "Only act while /control_mode equals control_mode_active_value."),
    ("auto_start", "false", "Start acting immediately without waiting for /control_mode."),
    ("publish_control_mode", "false", "Also run control_mode_publisher (simulation convenience)."),
    ("save_exec_trace", "false", "Save the execution trace as npz + png."),
    ("exec_trace_dir", "/home/hsr/deploy_record", "Directory for execution traces."),
    ("node_name", "hsr_openpi", "Node name."),
)


def _launch_setup(context, *args, **kwargs):
    profile = LaunchConfiguration("robot_profile").perform(context)
    share = get_package_share_directory("hsr_openpi")
    topics_yaml = os.path.join(share, "config", f"{profile}_topics.yaml")
    if not os.path.exists(topics_yaml):
        raise RuntimeError(f"Unknown robot_profile '{profile}' (expected config file {topics_yaml})")

    def cfg(name):
        return LaunchConfiguration(name)

    parameters = [
        topics_yaml,
        {
            "use_sim_time": cfg("use_sim_time"),
            "policy_backend": cfg("policy_backend"),
            "policy_host": cfg("policy_host"),
            "policy_port": cfg("policy_port"),
            "policy_connect_timeout": cfg("policy_connect_timeout"),
            "config_name": cfg("config_name"),
            "config_yaml": cfg("config_yaml"),
            "checkpoint_dir": cfg("checkpoint_dir"),
            "instruction": cfg("instruction"),
            "update_freq": cfg("update_freq"),
            "adopted_action_chunks": cfg("adopted_action_chunks"),
            "upsample": cfg("upsample"),
            "upsample_hz": cfg("upsample_hz"),
            "upsample_method": cfg("upsample_method"),
            "action_smoothing": cfg("action_smoothing"),
            "ema_alpha": cfg("ema_alpha"),
            "ma_window": cfg("ma_window"),
            "smooth_gripper": cfg("smooth_gripper"),
            "smooth_base": cfg("smooth_base"),
            "gripper_mode": cfg("gripper_mode"),
            "policy_image_order": cfg("policy_image_order"),
            "require_control_mode": cfg("require_control_mode"),
            "auto_start": cfg("auto_start"),
            "save_exec_trace": cfg("save_exec_trace"),
            "exec_trace_dir": cfg("exec_trace_dir"),
        },
    ]

    openpi_node = Node(
        package="hsr_openpi",
        executable="hsr_openpi_node",
        name=LaunchConfiguration("node_name"),
        output="screen",
        emulate_tty=True,
        parameters=parameters,
    )

    control_mode_node = Node(
        package="hsr_openpi",
        executable="control_mode_publisher",
        name="control_mode_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("publish_control_mode")),
        parameters=[{"use_sim_time": cfg("use_sim_time"), "mode": "auto"}],
    )

    return [openpi_node, control_mode_node]


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=desc) for n, d, desc in _ARGS]
    return LaunchDescription(declared + [OpaqueFunction(function=_launch_setup)])
