"""
Converts rosbags to numpy files for training.
"""
import os
import sys
current_script_path = os.path.abspath(__file__)
conversion_dir_path = os.path.dirname(current_script_path)
pkg_root_path = os.path.dirname(conversion_dir_path)
src_dir_path = os.path.join(pkg_root_path, 'collection', 'src')
sys.path.append(src_dir_path)

import pickle
import argparse

import cv2
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from rlds.tfds.episode_writer import EpisodeWriter
from rlds.rlds_types import build_episode, build_step

import hsr_utils as hsr


def load_images(traj_dir, name, transform=None):
    image_dir = os.path.join(traj_dir, name)
    images = []

    for imname in sorted(os.listdir(image_dir)):
        impath = os.path.join(image_dir, imname)
        # todo: proper preprocessing
        image = cv2.imread(impath)[:, :, ::-1]
        if transform:
            image = transform(image)
        images.append(image)

    return np.asarray(images, dtype=np.uint8)


def write_trajectory(writer, data):
    for instruciton in data["instructions"][0]:
        steps = []
        ep_len = len(data["actions"])
        for i in range(ep_len):
            step = build_step(
                dict(
                    head=data["image_head"][i],
                    hand=data["image_hand"][i],
                    state=data["qpos"][i],
                ),
                data["actions"][i],
                0,
                0,
                i == ep_len - 1,
                i == 0,
                i == ep_len - 1,
                {"language_instruction": instruciton}
            )
            steps.append(step)
        ep = build_episode(steps, {})
        writer.add_episode(ep)


def convert_dir(root, dirpath, writer):
    print(f"converting {dirpath}")

    with open(os.path.join(dirpath, "trajectory.pkl"), "rb") as f:
        traj_data = pickle.load(f)

    if traj_data["segments"][0]["has_suboptimal"] or not traj_data["segments"][0]["is_directed"]:
        print("skipping")
        return

    # transform_head_params = dict()
    # transform_hand_params = dict(
    #     rotate=90,
    #     K=traj_data["camera_info_hand"]["K"],
    #     D=traj_data["camera_info_hand"]["D"],
    # )
    transform_head_params = hsr.DEFAULT_TRANSFORM_HEAD
    transform_hand_params = hsr.DEFAULT_TRANSFORM_HAND
    transform_head = hsr.create_transform(**transform_head_params)
    transform_hand = hsr.create_transform(**transform_hand_params)

    traj_data["transform_head_params"] = transform_head_params
    traj_data["transform_hand_params"] = transform_hand_params

    traj_data["image_head"] = load_images(dirpath, "image_head", transform_head)
    traj_data["image_hand"] = load_images(dirpath, "image_hand", transform_hand)
    
    write_trajectory(writer, traj_data)


def get_dirs(root):
    paths = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        if "image_head" not in dirnames:
            continue

        paths.append(dirpath)

    return paths


def prepare_writer(outdir, max_episodes_per_file):
    ds_config = tfds.rlds.rlds_base.DatasetConfig(
        name='hsr_dataset',
        observation_info={
            "head": tfds.features.Image(
                shape=(224, 224, 3),
                encoding_format='jpeg'),
            "hand": tfds.features.Image(
                shape=(224, 224, 3),
                encoding_format='jpeg'),
            "state": tfds.features.Tensor(
                shape=(6,),
                dtype=tf.float32
            ),
        },
        action_info=tfds.features.Tensor(
            shape=(11,),
            dtype=tf.float32
        ),
        reward_info=tfds.features.Scalar(np.float32),
        discount_info=tfds.features.Scalar(np.float32),
        step_metadata_info={'language_instruction': tfds.features.Text()}
    )
    writer = EpisodeWriter(outdir, ds_config, max_episodes_per_file=max_episodes_per_file)
    return writer

def calc_output_num_episodes(dirs):
    num_episodes = 0
    for dirpath in dirs:
        with open(os.path.join(dirpath, "trajectory.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        num_episodes += len(traj_data["instructions"][0])
    return num_episodes
        

def convert(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    dirs = get_dirs(args.root)
    print(f"{len(dirs)} trajectories found")
    output_num_episodes = calc_output_num_episodes(dirs)
    writer = prepare_writer(args.outdir, min(100, output_num_episodes))

    params = [(args.root, x, writer) for x in dirs]

    for param in params:
        convert_dir(*param)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        help="",
    )
    parser.add_argument("outdir", type=str, help="output directory")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    convert(args)


if __name__ == "__main__":
    main()
