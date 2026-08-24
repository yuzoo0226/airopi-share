#!/usr/bin/env python3
"""Record a ROS 2 bag while a scripted driver moves the HSR in Gazebo.

Two drivers are available:

``pick``    (default) ``hsr_pick_task`` - spawns one of ten objects at a random
            pose on a table and performs a scripted top-down pick. This is the
            demonstration data the policy is trained on.
``random``  ``hsr_random_motion`` - a clamped Ornstein-Uhlenbeck random walk over
            every joint; useful for smoke testing the pipeline.

The launch also starts:

* ``image_transport republish`` for both cameras, so the bag carries the
  ``.../compressed`` topic names the real robot records (and is ~30x smaller);
* a ``ros_gz_bridge`` for Gazebo's ``dynamic_pose`` stream, which the pick task
  needs to see where the object actually is;
* ``ros2 bag record``.

The driver exits when it has finished its episodes and an OnProcessExit handler
tears the launch down, which sends SIGINT to the recorder - rosbag2 only writes
metadata.yaml on SIGINT.

Example
-------
    ros2 launch hsr_openpi collect_data.launch.py driver:=pick \
        bag_path:=/home/hsr/hsr_ros2_ws/_bags/pick_001 num_episodes:=100 seed:=1
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
GZ_POSE_TOPIC = "/gz/dynamic_pose"

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
    "/hsr_pick_task/episode",
    "/hsr_random_motion/episode",
    GZ_POSE_TOPIC,
]

_ARGS = (
    ("driver", "pick", "Scripted driver: 'pick' or 'random'."),
    ("bag_path", "/home/hsr/hsr_ros2_ws/_bags/pick", "Output bag directory (must not exist)."),
    ("storage", "mcap", "rosbag2 storage plugin."),
    ("num_episodes", "10", "Episodes to record (0 = until stopped)."),
    ("seed", "0", "Random seed."),
    ("record", "true", "Set false to drive without recording."),
    ("use_sim_time", "true", "Use the /clock published by Gazebo."),
    ("world", "default", "Ignition world name."),
    # pick driver
    ("close_time", "2.2", "Seconds spent closing the gripper (includes the grasp fix respawn)."),
    ("objects", "", "Comma separated subset of the object library (empty = all ten)."),
    ("results_path", "", "Where the per-episode success log is written."),
    # random driver
    ("episode_duration", "30.0", "Random driver: seconds of motion per episode."),
    ("reset_duration", "6.0", "Random driver: seconds spent returning to the start pose."),
    ("update_freq", "10.0", "Command rate [Hz]."),
    ("task", "move the arm around randomly", "Random driver: task string."),
)


def _launch_setup(context, *args, **kwargs):
    cfg = LaunchConfiguration
    bag_path = cfg("bag_path").perform(context)
    storage = cfg("storage").perform(context)
    driver = cfg("driver").perform(context).lower()
    world = cfg("world").perform(context)
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

    # Ground-truth model poses: the pick task uses them to place objects and to
    # judge whether one was actually lifted.
    actions.append(
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="gz_pose_bridge",
            output="log",
            # ros_gz_bridge names the ROS side after the Gazebo topic; the ROS
            # remap has to use the fully qualified name it actually creates.
            arguments=[f"/world/{world}/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V"],
            remappings=[(f"/world/{world}/dynamic_pose/info", GZ_POSE_TOPIC)],
            parameters=[{"use_sim_time": cfg("use_sim_time")}],
        )
    )

    if driver == "pick":
        objects = [o for o in cfg("objects").perform(context).split(",") if o]
        params = {
            "use_sim_time": cfg("use_sim_time"),
            "num_episodes": cfg("num_episodes"),
            "seed": cfg("seed"),
            "world": cfg("world"),
            "close_time": cfg("close_time"),
            "results_path": cfg("results_path"),
            "object_pose_topic": GZ_POSE_TOPIC,
        }
        if objects:
            params["objects"] = objects
        driver_node = Node(
            package="hsr_openpi",
            executable="pick_task",
            name="hsr_pick_task",
            output="screen",
            emulate_tty=True,
            parameters=[params],
        )
    else:
        driver_node = Node(
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
                    "task": cfg("task"),
                }
            ],
        )
    actions.append(driver_node)
    actions.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=driver_node,
                on_exit=[EmitEvent(event=Shutdown(reason="collection finished"))],
            )
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
