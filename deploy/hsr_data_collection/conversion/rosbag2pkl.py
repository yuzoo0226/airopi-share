"""
Converts rosbags to numpy files for training.
"""
import sys
import os
current_script_path = os.path.abspath(__file__)
conversion_dir_path = os.path.dirname(current_script_path)
pkg_root_path = os.path.dirname(conversion_dir_path)
src_dir_path = os.path.join(pkg_root_path, 'collection', 'src')
sys.path.append(src_dir_path)

import pickle
import argparse
from multiprocessing import Pool
from copy import deepcopy
import shutil
import json
import fnmatch

import rospy
import rosbag
import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R

from tf_bag import BagTfTransformer
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
from std_msgs.msg import Header
from cv_bridge import CvBridge

import hsr_utils as hsr
import trajectory_filter


def extract_initial_action(msgs):
    """
    Get initial values for action in trajectory
    """

    initial_joints = None

    for topic, msg, _ in msgs:
        if topic == hsr.JOINTS_TOPIC:
            initial_joints = msg
            break

    if initial_joints is None:
        raise ValueError("no joints topic")

    # return vector corresponding to ACT_KEYS
    res = dict()
    for name in hsr.ACTION_NAMES:
        value = 0

        if "base_" in name:  # x y or t
            value = 0
        elif name == "gripper":
            value = float(
                initial_joints.position[initial_joints.name.index(
                    "hand_motor_joint")]
                > 0
            )
        else:
            value = initial_joints.position[initial_joints.name.index(name)]

        res[name] = value

    return res


def append_bag_tf(bag_tf):
    """
    rosbag might be missing these TFs.
    Add these as static TFs.
    """

    transforms = [
        (
            (0, 0, 0),
            (0, 0, 0, 1),
            "base_link",
            "base_footprint"),
        (
            (0, 0, 0.1),
            (1, 0, 0, 0),
            "wrist_roll_link",
            "wrist_ft_sensor_frame"),
        (
            (0, 0, 0.155),
            (0, 0, 1, 0),
            "hand_palm_link",
            "wrist_roll_link"),
        (
            (-0.0786, 0.022, 0.248),
            (0.5, -0.5, 0.5, -0.5),
            "head_rgbd_sensor_link",
            "head_tilt_link"),
        (
            (0.026, 0, 0.0034),
            (0, 0, 0, 1),
            "hand_camera_frame",
            "hand_palm_link")
    ]

    tfs = []
    for tf in transforms:
        trans = tf[0]
        rot = tf[1]

        tf_msg = TransformStamped(
            header=Header(stamp=rospy.Time(1), frame_id=tf[3]),
            child_frame_id=tf[2],
            transform=Transform(
                translation=Vector3(x=trans[0], y=trans[1], z=trans[2]),
                rotation=Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3]),
            )
        )

        tfs.append(tf_msg)

    new_static_tfs = bag_tf.tf_static_messages + tfs
    bag_tf.tf_static_messages = sorted((tm for tm in new_static_tfs), key=lambda tfm: tfm.header.stamp.to_nsec())

    return bag_tf


# used to detect off-sync times
MAX_TIME_ERROR = 3


def get_msg_time(bag_msg, use_rosbag_times):
    topic, msg, bag_t = bag_msg

    if use_rosbag_times:
        return bag_t.to_sec()

    if topic in [hsr.CONTROL_TOPIC, hsr.BASE_TOPIC]:  # these topics do not have headers
        return bag_t.to_sec()

    t = msg.header.stamp.to_sec()

    if t == 0:
        # arm and head topic have stamp 0 due to mistake in old collection interface code
        if topic in [hsr.GRIPPER_CLOSE_TOPIC, hsr.GRIPPER_OPEN_TOPIC, hsr.ARM_TOPIC, hsr.HEAD_TOPIC]:
            return bag_t.to_sec()

        raise ValueError(f"header stamp = 0 for: {topic}")
    elif topic not in [hsr.BASE_TOPIC2, hsr.ARM_TOPIC]:  # todo: why these topics?
        error = abs(t - bag_t.to_sec())
        if error > MAX_TIME_ERROR:
            raise ValueError(f"time is very off ({topic}; header:{t}, bag:{bag_t.to_sec()}, error: {t - bag_t.to_sec()}); use --use-rosbag-times?")

    return t


def extract_msgs(bagpath, start_time, end_time, use_rosbag_times, max_traj_len_secs=-1):
    bag = rosbag.Bag(bagpath)

    topics = {
        hsr.ARM_TOPIC,
        hsr.GRIPPER_CLOSE_TOPIC,
        hsr.GRIPPER_OPEN_TOPIC,
        hsr.BASE_TOPIC,
        hsr.BASE_TOPIC2,
        hsr.HEAD_TOPIC,
        hsr.JOINTS_TOPIC,
        hsr.IMAGE_HEAD_TOPIC,
        hsr.IMAGE_HAND_TOPIC,
        hsr.CONTROL_TOPIC,
    }

    msgs = list(bag.read_messages(
        topics=topics, 
        start_time=rospy.Time(start_time) if start_time != -1 else None, 
        end_time=rospy.Time(end_time) if end_time != -1 else None)
    )

    msg_times = []

    for bag_msg in msgs:
        t = get_msg_time(bag_msg, use_rosbag_times)
        msg_times.append(t)

    valid = [i for i in range(len(msg_times)) if msg_times[i] is not None]
    msg_times = [msg_times[i] for i in valid]
    msgs = [msgs[i] for i in valid]

    if len(msg_times) == 0:
        raise ValueError("no valid bag messages")

    # replace original time and sort
    sorted_idx = np.argsort(msg_times)
    new_msgs = []
    for i in sorted_idx:
        topic, msg, _ = msgs[i]
        new_msgs.append((topic, msg, msg_times[i]))
    msgs = new_msgs

    msg_start_time = np.min(msg_times)
    msg_end_time = np.max(msg_times)

    # some bagfiles had a few messages many seconds earlier for some reason. time synchronization?
    # filter out messages that are more than N seconds older than last message
    if max_traj_len_secs > 0:
        msgs = [m for m in msgs if msg_end_time - m[2] < max_traj_len_secs]
        msg_times = [m[2] for m in msgs]

    msg_start_time = np.min(msg_times)
    msg_end_time = np.max(msg_times)

    return msgs, msg_start_time, msg_end_time


def get_pose_matrix(bag_tf, frame, ref_frame, lookup_time):
    curr_pose = bag_tf.lookupTransform(frame, ref_frame, lookup_time)

    curr_pos, curr_orn = curr_pose
    curr_mat = np.eye(4)
    curr_mat[:3, 3] = curr_pos
    curr_mat[:3, :3] = R.from_quat(curr_orn).as_matrix()

    return curr_mat


def calculate_base_velocity(msg, bag_tf):
    goal_x, goal_y, goal_t = msg.points[0].positions
    goal_mat = np.eye(4)
    goal_mat[0, 3] = goal_x
    goal_mat[1, 3] = goal_y
    goal_mat[:3, :3] = R.from_euler("z", goal_t).as_matrix()

    lookup_time = msg.header.stamp
    curr_mat = get_pose_matrix(bag_tf, "odom", "base_link", lookup_time)

    rel_mat = np.linalg.inv(curr_mat).dot(goal_mat)

    vel_x = rel_mat[0, 3] / 0.5
    vel_y = rel_mat[1, 3] / 0.5
    vel_t = R.from_matrix(rel_mat[:3, :3]).as_euler("xyz")[2] / 0.5

    return vel_x, vel_y, vel_t


def extract_single_trajectory(bagpath, start_time, end_time, timestep, use_rosbag_times, max_traj_len_secs):
    res = extract_msgs(bagpath, start_time, end_time, use_rosbag_times, max_traj_len_secs)
    
    if res is None:
        raise ValueError("unable to extract ros messages")

    msgs, msg_start_time, msg_end_time = res
    initial_act = extract_initial_action(msgs)

    if initial_act is None:
        raise ValueError("no action data")

    bridge = CvBridge()

    # entire trajectory
    data = {
        "image_head": [],
        "image_hand": [],
        "qpos": [],
        "qvel": [],
        "actions": [],
        "human_control": [],
        "time_stamp": [],
        "base_to_eef_transform": [],
        "base_to_hand_camera_transform": [],
        "base_to_head_camera_transform": [],
        "odom_to_base_transform": [],
    }

    # observation values of a single step
    obs = {
        "image_head": None,
        "image_hand": None,
        "qpos": None,
        "qvel": None,
        "human_control": True,  # default as True

        "base_to_eef_transform": None,
        "base_to_hand_camera_transform": None,
        "base_to_head_camera_transform": None,
        "odom_to_base_transform": None,
    }

    # action values of a single step
    act = initial_act

    start_demo = True  # start conversion once action topics start

    # obs_times = []

    os.makedirs("/tmp", exist_ok=True)
    tmp_bag_path = f"/tmp/bag_{msg_start_time}_{msg_end_time}.bag"
    with rosbag.Bag(tmp_bag_path, "w") as tmp_bag:
        for topic, msg, t in rosbag.Bag(bagpath).read_messages(
            topics=[], # all topics
            start_time=rospy.Time(start_time) if start_time != -1 else None, 
            end_time=rospy.Time(end_time) if end_time != -1 else None
        ):
            tmp_bag.write(topic, msg, t)
    bag_tf = BagTfTransformer(tmp_bag_path)
    bag_tf = append_bag_tf(bag_tf)
    os.remove(tmp_bag_path)

    msg_idx = 0
    target_time = msg_start_time

    num_total_frames = 0
    num_skipped_frames = 0
    header_time = -1

    while target_time < msg_end_time:
        last_msg_time = None

        # if it's been longer than N seconds since the base is moved, assume mobile base velocity 0
        last_base_msg_time = None
        action_recorded = None
        joints_moving = False

        while msg_idx < len(msgs):
            topic, msg, msg_time_s = msgs[msg_idx]

            # quick fix
            if topic not in [hsr.CONTROL_TOPIC, hsr.BASE_TOPIC, hsr.GRIPPER_CLOSE_TOPIC, hsr.GRIPPER_OPEN_TOPIC, hsr.ARM_TOPIC, hsr.HEAD_TOPIC]:  # these topics do not have headers
                header_time = msg.header.stamp.to_sec()

            if msg_time_s > target_time and start_demo:
                break

            msg_idx += 1
            last_msg_time = msg_time_s

            if topic == hsr.ARM_TOPIC or topic == hsr.HEAD_TOPIC:
                assert len(msg.points) == 1

                for joint_name, pos in zip(msg.joint_names, msg.points[0].positions):
                    act[joint_name] = pos

                action_recorded = "arm/head"
            elif topic == hsr.GRIPPER_CLOSE_TOPIC:
                act["gripper"] = 0
                action_recorded = "gripper"
            elif topic == hsr.GRIPPER_OPEN_TOPIC:
                act["gripper"] = 1
                action_recorded = "gripper"
            elif topic == hsr.BASE_TOPIC:
                act["base_x"] = msg.linear.x
                act["base_y"] = msg.linear.y
                act["base_t"] = msg.angular.z

                last_base_msg_time = msg_time_s
                action_recorded = "base"
            elif topic == hsr.BASE_TOPIC2:
                last_base_msg_time = msg_time_s

                if len(msg.points) == 0:
                    # base stop message
                    action_recorded = "base2 (stop)"
                    act["base_x"] = 0
                    act["base_y"] = 0
                    act["base_t"] = 0
                    
                    continue

                try:
                    vel_x, vel_y, vel_t = calculate_base_velocity(msg, bag_tf)
                except RuntimeError as e:
                    if "Could not find the transformation" not in str(e):
                        print("base command:", e)
                    continue

                action_recorded = "base2"
                act["base_x"] = vel_x
                act["base_y"] = vel_y
                act["base_t"] = vel_t
            elif topic in [
                hsr.IMAGE_HAND_TOPIC,
                hsr.IMAGE_HEAD_TOPIC,
            ]:
                # img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                np_arr = np.fromstring(msg.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, ::-1]

                key_name = "image_head" if topic == hsr.IMAGE_HEAD_TOPIC else "image_hand"
                obs[key_name] = img

                # sync TF with image
                try:
                    # print(bag_tf.getTransformGraphInfo())
                    tf_time = rospy.Time(header_time if header_time > 0 else msg_time_s)
                    obs["odom_to_base_transform"] = get_pose_matrix(bag_tf, "odom", "base_link", tf_time)
                    obs["base_to_eef_transform"] = get_pose_matrix(bag_tf, "base_link", "hand_palm_link", tf_time)
                    obs["base_to_hand_camera_transform"] = get_pose_matrix(bag_tf, "base_link", "hand_camera_frame", tf_time)
                    obs["base_to_head_camera_transform"] = get_pose_matrix(bag_tf, "base_link", "head_rgbd_sensor_link", tf_time)
                except RuntimeError as e:
                    if "Could not find the transformation" not in str(e):
                        pass
                    continue
            elif topic == hsr.JOINTS_TOPIC:
                obs["qpos"] = [msg.position[msg.name.index(name)] for name in hsr.JOINT_NAMES]
                obs["qvel"] = [msg.velocity[msg.name.index(name)] for name in hsr.JOINT_NAMES]

                joints_moving = np.any(np.abs(np.asarray(msg.velocity)) > 0.1)
            elif topic == hsr.CONTROL_TOPIC:
                obs["human_control"] = msg.data != "model"

        start_demo |= (action_recorded is not None)

        # velocity type actions are 0 if no messages are sent this timestep
        if last_base_msg_time is None or (target_time - last_base_msg_time) > timestep:
            act["base_x"] = 0
            act["base_y"] = 0
            act["base_t"] = 0

        num_total_frames += 1
        target_time += timestep

        skip = action_recorded is None and not joints_moving

        # dont add any data until observation is complete
        if any([v is None for v in obs.values()]) or skip:
            num_skipped_frames += 1
            # print("SKIP")
            continue

        # cv2.imshow("rgb", obs["image_hand"][:, :, ::-1])
        # print(start_demo, action_recorded, joints_moving)
        # cv2.waitKey(0)

        data["time_stamp"].append(float(target_time))

        for k in obs.keys():
            data[k].append(deepcopy(obs[k]))

        data["actions"].append([act[k] for k in hsr.ACTION_NAMES])
        # obs_times.append(last_msg_time or target_time)

    return data

def list_depth(lst):
    if not isinstance(lst, list):
        return 0  # リストではない場合は深さ0
    elif not any(isinstance(i, list) for i in lst):
        return 1  # リストだが中にリストがない場合は深さ1（シングルリスト）
    else:
        return 1 + max(list_depth(i) for i in lst if isinstance(i, list))
    
def bag2arrays(
    bagpath,
    meta_data,
    img_transform=None,
    timestep=0.1,
    use_rosbag_times=False,
    max_traj_len_secs=-1,
    clip_top_down_grasps=False,
    large_object=False,
    num_workers=1
):
    trajectories = []
    meta_data_batch = []
    segments = meta_data["segments"]

    if len(segments) > 1 and num_workers > 1:
        pool_args = [
            (bagpath, segment, timestep, use_rosbag_times, max_traj_len_secs)
            for segment in segments
        ]

        with Pool(processes=num_workers) as pool:
            results = pool.map(extract_single_trajectory_wrapper, pool_args)

        for raw_data, segment in results:
            if clip_top_down_grasps:
                raise NotImplementedError("clip_top_down_grasps is commented out because it may badly affect conversion")
                # trajectories = trajectory_filter.clip_top_down_grasps(raw_data, large_object=large_object)
            else:
                trajectories.append(raw_data)
                meta_data_i = deepcopy(meta_data)
                meta_data_i["segments"] = [segment]
                l_depth = list_depth(meta_data["instructions"])
                if l_depth == 1:
                    if "instruction_index" in segment.keys():
                        meta_data_i["instructions"] = [[meta_data["instructions"][segment["instruction_index"]]]]
                    elif "instructions_index" in segment.keys():
                        meta_data_i["instructions"] = [[meta_data["instructions"][segment["instructions_index"]]]]
                elif l_depth == 2:
                    if "instruction_index" in segment.keys():
                        meta_data_i["instructions"] = [meta_data["instructions"][segment["instruction_index"]]]
                    elif "instructions_index" in segment.keys():
                        meta_data_i["instructions"] = [meta_data["instructions"][segment["instructions_index"]]]
                else:
                    raise ValueError("invalid instructions list depth")
                meta_data_batch.append(meta_data_i)
    else:
        for segment in segments:
            raw_data = extract_single_trajectory(
                bagpath, 
                segment["start_time"],
                segment["end_time"],
                timestep, use_rosbag_times, max_traj_len_secs
            )

            if clip_top_down_grasps:
                raise NotImplementedError("clip_top_down_grasps is commented out because it may badly affect conversion")
                # trajectories = trajectory_filter.clip_top_down_grasps(raw_data, large_object=large_object)
            else:
                trajectories.append(raw_data)
                meta_data_i = deepcopy(meta_data)
                meta_data_i["segments"] = [segment]
                l_depth = list_depth(meta_data["instructions"])
                if l_depth == 1:
                    if "instruction_index" in segment.keys():
                        meta_data_i["instructions"] = [[meta_data["instructions"][segment["instruction_index"]]]]
                    elif "instructions_index" in segment.keys():
                        meta_data_i["instructions"] = [[meta_data["instructions"][segment["instructions_index"]]]]
                elif l_depth == 2:
                    if "instruction_index" in segment.keys():
                        meta_data_i["instructions"] = [meta_data["instructions"][segment["instruction_index"]]]
                    elif "instructions_index" in segment.keys():
                        meta_data_i["instructions"] = [meta_data["instructions"][segment["instructions_index"]]]
                else:
                    raise ValueError("invalid instructions list depth")
                meta_data_batch.append(meta_data_i)

    if img_transform is None:
        img_transform = lambda x, head_image: x
    else:
        raise NotImplementedError("img_transform is disabled because it may badly affect conversion")

    trajectory_batch = []
    for traj_data in trajectories:
        # preprocess images
        traj_data["image_head"] = np.asarray(img_transform(traj_data["image_head"], head_image=True))
        traj_data["image_hand"] = np.asarray(img_transform(traj_data["image_hand"], head_image=False))

        # add extra information
        demo_len = len(traj_data["actions"])
        traj_data["length"] = demo_len
        traj_data["joint_names"] = hsr.JOINT_NAMES
        traj_data["action_names"] = hsr.ACTION_NAMES

        trajectory_batch.append(traj_data)

    return trajectory_batch, meta_data_batch


def extract_single_trajectory_wrapper(args):
    bagpath, segment, timestep, use_rosbag_times, max_traj_len_secs = args
    print(f"  {segment['start_time']} - {segment['end_time']}")
    raw_data = extract_single_trajectory(
        bagpath, 
        segment["start_time"],
        segment["end_time"],
        timestep, use_rosbag_times, max_traj_len_secs
    )
    return raw_data, segment


def write_images(traj_dir, name, trajectory):
    image_dir = os.path.join(traj_dir, name)
    os.makedirs(image_dir, exist_ok=True)

    for i, image in enumerate(trajectory[name]):
        cv2.imwrite(os.path.join(image_dir, f"{i:06d}.jpg"), image[:, :, ::-1])


def write_trajectory(traj_dir, traj_data):
    write_images(traj_dir, "image_head", traj_data)
    write_images(traj_dir, "image_hand", traj_data)

    states = dict()

    states["camera_info_head"] = traj_data.get("camera_info_head", hsr.DEFAULT_CAMERA_INFO_HEAD)
    states["camera_info_hand"] = traj_data.get("camera_info_head", hsr.DEFAULT_CAMERA_INFO_HAND)
    states["time_stamp"] = traj_data["time_stamp"]

    states["joint_names"] = traj_data["joint_names"]
    states["qpos"] = np.float32(traj_data["qpos"])
    states["qvel"] = np.float32(traj_data["qvel"])

    states["action_names"] = traj_data["action_names"]
    states["actions"] = np.float32(traj_data["actions"])

    for k, v in traj_data.items():
        if "image" in k:
            continue
        elif k.endswith("_transform"):
            states[k] = np.float32(traj_data[k])
        elif k not in states:
            states[k] = v

    # for k, v in states.items():
    #     print(k, type(v))

    #     if isinstance(v, list):
    #         print(type(v[0]))

    #         if isinstance(v[0], list):
    #             print(type(v[0][0]))

    with open(os.path.join(traj_dir, "trajectory.pkl"), 'wb') as f:
        pickle.dump(states, f)


def find_closest_meta_json(subdir):
    current_dir = os.path.abspath(subdir)
    root_dir = os.path.abspath(os.sep)

    while current_dir != root_dir:
        meta_json_path = os.path.join(current_dir, 'meta.json')
        if os.path.isfile(meta_json_path):
            return meta_json_path
        current_dir = os.path.dirname(current_dir)

    raise ValueError(f"Could not find meta.json in any ancestor: {subdir}. Create a meta.json.")

def match_path_in_meta_json(subdir, meta_json_path):
    meta_json_dir = os.path.dirname(meta_json_path)

    try:
        with open(meta_json_path, 'r') as file:
            try:
                data = json.load(file)
            except json.decoder.JSONDecodeError as e:
                raise ValueError(f"json error: {meta_json_path}, {e}")
            relative_subdir = os.path.relpath(subdir, meta_json_dir)
            for obj in data:
                path_pattern = obj.get('bag_path', '')
                if fnmatch.fnmatch(relative_subdir, path_pattern):
                    return obj
    except Exception as e:
        raise ValueError(f"error: {meta_json_path}, {e}")

    raise ValueError(f"Did not find meta.json with matching path: {subdir}")


def get_bag_files(root):
    paths = []

    for dirpath, dirnames, filenames in os.walk(root):
        for file in filenames:
            if file.endswith('.bag'):
                paths.append(os.path.join(dirpath, file))

    return sorted(paths)

def process_bag_single_segment(args, bagpath, meta_data):
    root = args.root
    outdir = args.outdir

    dirpath = os.path.dirname(bagpath)
    relative_dir = os.path.relpath(dirpath, root)
    target_dir = os.path.join(outdir, relative_dir)
    bagname = os.path.splitext(os.path.basename(bagpath))[0]

    if not args.overwrite and os.path.exists(os.path.join(target_dir, bagname)):
        print(f"skip: {target_dir} {bagname}")
        return

    try:
        print(f"extracting: {bagpath}")

        trajectory_batch, meta_data_batch = bag2arrays(
            bagpath,
            meta_data,
            timestep=1 / args.hz,
            use_rosbag_times=args.use_rosbag_times,
            max_traj_len_secs=args.max_length,
            clip_top_down_grasps=args.clip_top_down_grasps,
            large_object=args.large_object,
            num_workers=1,
        )

        print(f"finished extracting: {bagpath}")
    except ValueError as e:
        print("failed to extract data:", bagpath, str(e))
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    for traj_i, (traj_data, meta_data_) in enumerate(zip(trajectory_batch, meta_data_batch)):
        traj_data.update(meta_data_)
        traj_data.update(dict(
            use_rosbag_times=args.use_rosbag_times,
            clip_top_down_grasps=args.clip_top_down_grasps,
        ))

        traj_dir = os.path.join(target_dir, bagname, f"{traj_i:03d}")
        if not args.overwrite and os.path.exists(traj_dir):
            print("skip:", traj_dir)
            continue

        print(f"creating: {traj_dir}")
        os.makedirs(traj_dir, exist_ok=True)

        write_trajectory(traj_dir, traj_data)


def process_bag_multi_segment(args, bagpath, meta_data):
    root = args.root
    outdir = args.outdir

    dirpath = os.path.dirname(bagpath)
    relative_dir = os.path.relpath(dirpath, root)
    target_dir = os.path.join(outdir, relative_dir)
    bagname = os.path.splitext(os.path.basename(bagpath))[0]

    if not args.overwrite and os.path.exists(os.path.join(target_dir, bagname)):
        print(f"skip: {target_dir} {bagname}")
        return

    try:
        print(f"extracting: {bagpath}")

        trajectory_batch, meta_data_batch = bag2arrays(
            bagpath,
            meta_data,
            timestep=1 / args.hz,
            use_rosbag_times=args.use_rosbag_times,
            max_traj_len_secs=args.max_length,
            clip_top_down_grasps=args.clip_top_down_grasps,
            large_object=args.large_object,
            num_workers=args.num_workers,
        )

        print(f"finished extracting: {bagpath}")
    except ValueError as e:
        print("failed to extract data:", bagpath, str(e))
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    for traj_i, (traj_data, meta_data_) in enumerate(zip(trajectory_batch, meta_data_batch)):
        traj_data.update(meta_data_)
        traj_data.update(dict(
            use_rosbag_times=args.use_rosbag_times,
            clip_top_down_grasps=args.clip_top_down_grasps,
        ))

        traj_dir = os.path.join(target_dir, bagname, f"{traj_i:03d}")
        if not args.overwrite and os.path.exists(traj_dir):
            print("skip:", traj_dir)
            continue

        print(f"creating: {traj_dir}")
        os.makedirs(traj_dir, exist_ok=True)

        write_trajectory(traj_dir, traj_data)


def convert_files(args):
    bagpaths = get_bag_files(args.root)
    params_single_segment = []
    params_multi_segment = []

    print("checking wheather each rosbag contains multiple episodes or not...")
    for bagpath in bagpaths:
        dirpath = os.path.dirname(bagpath)
        meta_path = find_closest_meta_json(dirpath)
        meta_data = match_path_in_meta_json(dirpath, meta_path)
        num_segments = len(meta_data.get("segments", []))
        if num_segments > 1:
            print(f"  multiple episodes in {bagpath}")
            params_multi_segment.append((args, bagpath, meta_data))
        else:
            print(f"  single episode in {bagpath}")
            params_single_segment.append((args, bagpath, meta_data))
    
    print("-" * 50)

    if params_single_segment:
        with Pool(processes=args.num_workers) as pool:
            pool.starmap(process_bag_single_segment, params_single_segment)

    for param in params_multi_segment:
        process_bag_multi_segment(*param)


def has_jpg_files(directory):
    """Return True if directory has any .jpg files, False otherwise."""
    for _, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                return True
    return False

def delete_empty_directories(root_directory):
    """Delete directories without .jpg files."""
    # We want to traverse directories from the deepest level first.
    # This is done to ensure that we don't delete a parent directory before checking its children.
    for directory, _, _ in sorted(os.walk(root_directory), key=lambda x: x[0], reverse=True):
        if not has_jpg_files(directory):
            shutil.rmtree(directory, ignore_errors=True)
            print(f"Deleted: {directory}")
            

def convert(args):
    convert_files(args)
    delete_empty_directories(args.outdir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        help="",
    )
    parser.add_argument("outdir", type=str, help="output directory")

    parser.add_argument("--num-workers", type=int, default=4, help="numbers of processes to use")

    parser.add_argument("--hz", type=int, default=10, help="sample rate")
    parser.add_argument("--max-length", type=float, default=-1, help="maximum length of trajectory in seconds")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--use-rosbag-times",
        action="store_true",
        help="use rosbag times for all messages instead of header timestamps",
    )

    parser.add_argument(
        "--clip-top-down-grasps",
        action="store_true",
        help="use rule-based function that filters and segments data into individual top-down grasp trajectories",
    )
    parser.add_argument(
        "--large-object",
        action="store_true",
        help="used for top-down grasp filter",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    convert(args)


if __name__ == "__main__":
    main()
