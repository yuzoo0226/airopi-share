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
from multiprocessing import Pool

import cv2
import numpy as np

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


def convert_dir(root, dirpath, outdir, overwrite):
    traj_name = "-".join(dirpath.split("/")[-2:]) + ".npz"
    relative_dir = os.path.relpath(os.path.join(*dirpath.split("/")[:-2]), root)
    
    # Determine the target directory to create or use
    target_dir = os.path.join(outdir, relative_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    new_traj_path = os.path.join(target_dir, traj_name)

    if not overwrite and os.path.exists(new_traj_path):
        print(f"skipping: {new_traj_path}")
        return

    print(f"converting: {new_traj_path}")

    with open(os.path.join(dirpath, "trajectory.pkl"), "rb") as f:
        traj_data = pickle.load(f)

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
    
    with open(new_traj_path, "wb") as f:
        np.savez(f, **traj_data)


def get_dirs(root):
    paths = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        if "image_head" not in dirnames:
            continue

        paths.append(dirpath)

    return paths


def convert(args):
    dirs = get_dirs(args.root)
    print(f"{len(dirs)} trajectories found")

    params = [(args.root, x, args.outdir, args.overwrite) for x in dirs]

    with Pool(processes=args.num_workers) as pool:
        pool.starmap(convert_dir, params)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        help="",
    )
    parser.add_argument("outdir", type=str, help="output directory")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="",
    )
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    convert(args)


if __name__ == "__main__":
    main()
