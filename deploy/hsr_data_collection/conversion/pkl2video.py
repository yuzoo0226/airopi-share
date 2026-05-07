import os
import argparse
from pathlib import Path
import pickle
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

from moviepy.editor import ImageSequenceClip

import hsr_utils as hsr
from pkl2np import load_images, get_dirs


def convert_dir(root, traj_dir, outdir, overwrite):
    act_range = np.asarray(hsr.ACTION_RANGE)
    joint_range = np.asarray(hsr.JOINT_RANGE)
    bar_max_height = 50
    bar_width = 10
    img_height = 224
    text_color = (255, 0, 255)

    frames = []

    traj_name = "-".join(traj_dir.split("/")[-2:])
    relative_dir = os.path.relpath(os.path.join(*traj_dir.split("/")[:-2]), root)
    target_dir = os.path.join(outdir, relative_dir)
    video_path = os.path.join(target_dir, traj_name + ".mp4")

    if os.path.exists(video_path) and not overwrite:
        return

    with open(os.path.join(traj_dir, "trajectory.pkl"), "rb") as f:
        traj_data = pickle.load(f)

    transform = hsr.create_transform(image_size=224)

    image_head = load_images(traj_dir, "image_head", transform)
    image_hand = load_images(traj_dir, "image_hand", transform)

    joints = traj_data["qpos"]
    joints = (joints - joint_range[:, 0]) / (joint_range[:, 1] - joint_range[:, 0])

    actions = traj_data["actions"]
    actions = (actions - act_range[:, 0]) / (act_range[:, 1] - act_range[:, 0])

    for step_i, step_data in enumerate(zip(image_head, image_hand, joints, actions)):
        imhead, imhand, q, a = step_data

        frame = np.hstack([imhead, imhand])

        # text_traj = f"{traj_count}/{total_files}: {traj_name}"
        # frame = cv2.putText(frame, text_traj, (5, 10), cv2.FONT_HERSHEY_PLAIN, 0.5, text_color, 1)
        frame = cv2.putText(frame, traj_data["instructions"][0], (5, 20), cv2.FONT_HERSHEY_PLAIN, 0.5, text_color, 1)
        text_step = f"{step_i+1}/{len(image_head)}"
        frame = cv2.putText(frame, text_step, (5, 30), cv2.FONT_HERSHEY_PLAIN, 0.5, text_color, 1)
        text_misc = ""
        text_misc += "R " if traj_data["use_rosbag_times"] else ""
        text_misc += "C " if traj_data["clip_top_down_grasps"] else ""

        frame = cv2.putText(frame, text_misc, (5, 40), cv2.FONT_HERSHEY_PLAIN, 0.5, text_color, 1)

        for q_i, val in enumerate(q):
            if q_i == 0:
                color = (128, 128, 255)
                loc = 8
            else:
                color = (255, 128, 128)
                loc = q_i - 1

            bar_height = int(val * bar_max_height)
            start_point = (loc * bar_width, img_height - bar_height)
            end_point = ((loc + 1) * bar_width, img_height)
            cv2.rectangle(frame, start_point, end_point, color, 2)

        offset_x = 0

        for a_i, val in enumerate(a):
            if a_i < 5:
                color = (255, 0, 0)
            elif a_i < 8:
                color = (0, 255, 0)
            elif a_i == 8:
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)
            
            bar_height = int(val * bar_max_height)
            start_point = (a_i * bar_width + offset_x, img_height - bar_height)
            end_point = ((a_i + 1) * bar_width + offset_x, img_height)
            cv2.rectangle(frame, start_point, end_point, color, 1)
        
        frames.append(frame)

    clip = ImageSequenceClip(frames, fps=10)  # todo: no hardcode

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    clip.write_videofile(video_path, codec="libx264")


def convert(args):
    dirs = get_dirs(args.root)
    print(f"{len(dirs)} trajectories found")

    params = [(args.root, traj_dir, args.outdir, args.overwrite) for traj_dir in dirs]

    with Pool(processes=8) as pool:
        pool.starmap(convert_dir, params)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="root directory of pkls")
    parser.add_argument("outdir", type=str, help="output directory")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    convert(args)


if __name__ == "__main__":
    main()
