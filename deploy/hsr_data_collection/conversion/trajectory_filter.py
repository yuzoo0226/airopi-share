import os
import sys
current_script_path = os.path.abspath(__file__)
conversion_dir_path = os.path.dirname(current_script_path)
pkg_root_path = os.path.dirname(conversion_dir_path)
src_dir_path = os.path.join(pkg_root_path, 'collection', 'src')
sys.path.append(src_dir_path)

import numpy as np

import hsr_utils as hsr


def clip_top_down_grasps(data, large_object=False):
    act = np.asarray(data["actions"])
    joints = np.asarray(data["qpos"])

    state_lift = joints[:, hsr.JOINT_NAMES.index("arm_lift_joint")]
    act_lift = act[:, hsr.ACTION_NAMES.index("arm_lift_joint")]
    arm_change = act_lift - state_lift

    arm_going_down = arm_change < -0.001
    arm_going_up = arm_change > 0

    hand_joint = joints[:, hsr.JOINT_NAMES.index("hand_motor_joint")]
    hand_open = hand_joint > 1.0
    hand_closed = hand_joint < -0.6
    fingers_moved = [0.0] + np.abs(hand_joint[1:] - hand_joint[:-1])
    command_close = act[:, hsr.ACTION_NAMES.index("gripper")] == 0

    down = np.logical_and(hand_open, arm_going_down)

    if large_object:
        up = np.logical_and.reduce([command_close, arm_going_up, ~hand_closed])
    else:
        up = np.logical_and.reduce([command_close, arm_going_up])

    if np.sum(up) == 0:
        return []

    # filter out incomplete trajectories
    last_up = np.where(up)[0][-1]
    down[last_up:] = False

    trajectories = []
    key = list(data.keys())[0]
    current_trajectory = {k: [] for k in data.keys()}
    went_down, went_up = 0, 0
    cuts = 0

    # extract
    for i, (d, u, fm) in enumerate(zip(down, up, fingers_moved)):
        dropped = went_down >= 10
        lifted = went_up >= 10

        if lifted and dropped and not u:
            if len(current_trajectory[key]) > 0:
                for k, v in current_trajectory.items():
                    current_trajectory[k] = np.asarray(v)
                trajectories.append(current_trajectory)
                current_trajectory = {k: [] for k in data.keys()}

            went_up, went_down = 0, 0
            cuts = 0
        elif (u and dropped) or d or fm > 0.1:
            cuts = 0
            for k in data.keys():
                current_trajectory[k].append(data[k][i])
        else:
            cuts += 1

            if cuts >= 10:
                current_trajectory = {k: [] for k in data.keys()}
                went_up, went_down = 0, 0
        
        went_down += int(d)
        went_up += int(u)

    if len(current_trajectory[key]) > 0:
        trajectories.append(current_trajectory)

    return trajectories
