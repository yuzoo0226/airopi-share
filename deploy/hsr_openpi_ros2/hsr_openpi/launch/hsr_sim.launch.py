#!/usr/bin/env python3
"""Start the HSR Ignition Gazebo simulation, optionally headless.

``hsrb_gazebo_bringup`` builds the Gazebo command line as
``-r -v 3 <world_file_name>``; it exposes no switch for server-only or headless
rendering. Because ``world_file_name`` ends up verbatim in that string we can
prepend the extra ``gz sim`` flags to it, which is what ``headless:=true`` does
here (``-s`` = server only, ``--headless-rendering`` = render sensors through
EGL instead of GLX so no X server is needed).

Examples
--------
    # headless, empty world (default)
    ros2 launch hsr_openpi hsr_sim.launch.py

    # headless, apartment world, robot at the usual start pose
    ros2 launch hsr_openpi hsr_sim.launch.py world:=apartment_no_objects \
        robot_pos_x:=5.0 robot_pos_y:=6.6

    # with the Gazebo GUI (requires DISPLAY)
    ros2 launch hsr_openpi hsr_sim.launch.py headless:=false
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

_ARGS = (
    ("world", "empty", "World name inside tmc_gazebo_worlds, or an absolute .world/.sdf path."),
    ("headless", "true", "Run gz sim server-only with EGL sensor rendering (no X server needed)."),
    ("robot_pos_x", "0.0", "Initial robot x position."),
    ("robot_pos_y", "0.0", "Initial robot y position."),
    ("robot_pos_z", "0.0", "Initial robot z position."),
    ("robot_rpy_Y", "0.0", "Initial robot yaw."),
    ("robot_name", "hsrb", "Robot name (hsrb / hsrc)."),
)


def _resolve_world(name: str) -> str:
    if os.path.isabs(name):
        return name
    worlds_dir = os.path.join(get_package_share_directory("tmc_gazebo_worlds"), "worlds")
    for candidate in (name, f"{name}.world", f"{name}.sdf"):
        path = os.path.join(worlds_dir, candidate)
        if os.path.exists(path):
            return path
    available = sorted(f for f in os.listdir(worlds_dir) if f.endswith((".world", ".sdf")))
    raise RuntimeError(f"World '{name}' not found in {worlds_dir}. Available: {', '.join(available)}")


def _launch_setup(context, *args, **kwargs):
    world = _resolve_world(LaunchConfiguration("world").perform(context))
    headless = LaunchConfiguration("headless").perform(context).lower() in ("1", "true", "yes")
    robot_name = LaunchConfiguration("robot_name").perform(context)

    # Extra gz flags are smuggled in through world_file_name (see the docstring).
    world_file_name = f"--headless-rendering -s {world}" if headless else world

    bringup = os.path.join(
        get_package_share_directory("hsrb_gazebo_bringup"), "launch", f"{robot_name}_gazebo_bringup.launch.py"
    )
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={
                "world_file_name": world_file_name,
                "robot_name": robot_name,
                "robot_pos_x": LaunchConfiguration("robot_pos_x"),
                "robot_pos_y": LaunchConfiguration("robot_pos_y"),
                "robot_pos_z": LaunchConfiguration("robot_pos_z"),
                "robot_rpy_Y": LaunchConfiguration("robot_rpy_Y"),
            }.items(),
        )
    ]


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=desc) for n, d, desc in _ARGS]
    return LaunchDescription(declared + [OpaqueFunction(function=_launch_setup)])
