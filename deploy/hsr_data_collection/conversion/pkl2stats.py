import os
import argparse
from collections import defaultdict
from multiprocessing import Pool
import pickle
import csv
import random
from datetime import datetime, timedelta

import cv2
import pandas as pd
import numpy as np

from pathlib import Path
from moviepy.editor import ImageSequenceClip

from pkl2np import load_images, get_dirs


def collect_stats(dirpath, root):
    with open(os.path.join(dirpath, "trajectory.pkl"), "rb") as f:
        traj_data = pickle.load(f)

    return dict(
        # dirpath=dirpath,
        dirpath = os.path.relpath(dirpath, root),
        instruction=traj_data["instructions"][0],
        length=traj_data["length"],
        has_suboptimal=traj_data["segments"][0]["has_suboptimal"],
        is_directed=traj_data["segments"][0]["is_directed"],
        teleop_interface=traj_data["teleop_interface"],
        first_time_stamp=datetime.utcfromtimestamp(traj_data["time_stamp"][0]),
        last_time_stamp=datetime.utcfromtimestamp(traj_data["time_stamp"][-1]),
    )


def write_csv(headers, dictlist, outpath):
    with open(outpath, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for d in dictlist:
            writer.writerow(d)


def summarize_groups(df):
    # Extract the grandparent directory from 'dirpath'
    df['grandparent_dir'] = df['dirpath'].apply(lambda x: str(Path(x).parent.parent))

    summary = df.groupby(['grandparent_dir', 'instruction', 'has_suboptimal', 'is_directed', 'teleop_interface'], dropna=False) \
                .agg(
                    num_trajectories=('grandparent_dir', 'size'),
                    total_frames=('length', 'sum'),
                    average_frames=('length', 'mean'),
                    first_time_stamp=('first_time_stamp', 'min'),
                    last_time_stamp=('last_time_stamp', 'max'),
                ).reset_index()

    return summary


def convert(args):
    dirs = get_dirs(args.root)
    print(f"{len(dirs)} trajectories found")
    params = [(d, args.root) for d in dirs]

    with Pool(processes=32) as pool:
        result = pool.starmap(collect_stats, params)

    write_csv(["dirpath", "instruction", "length", "has_suboptimal", "is_directed", "teleop_interface", "first_time_stamp", "last_time_stamp"], result, args.out)

    df = pd.read_csv(args.out)
    summarize_groups(df).to_csv(os.path.splitext(args.out)[0] + "_groups.csv", index=False)


def generate_video(root, grandparent_dir):
    # choose a random trajectory to animate
    dpath = Path(root) / Path(grandparent_dir)
    dpath = dpath / Path(random.choice(os.listdir(dpath)))
    dpath = dpath / Path(random.choice(os.listdir(dpath)))

    image_head = load_images(str(dpath), "image_head")
    image_hand = load_images(str(dpath), "image_hand")
    images = list(np.concatenate([image_head, image_hand], 2))
    
    clip = ImageSequenceClip(images, fps=10)  # todo: remove hardcoding
    fname = grandparent_dir.replace("/", ".")
    # clip.write_gif(os.path.join("gifs", fname + ".gif"))
    vpath = os.path.join("mp4", fname + ".mp4")
    clip.write_videofile(vpath, audio=False)


def generate_sample_videos(args):
    csv_path = os.path.splitext(args.out)[0] + "_groups.csv"
    df = pd.read_csv(csv_path)
    params = [(args.root, dpath) for dpath in list(df['grandparent_dir'].unique())]

    # with Pool(processes=8) as pool:
    #     pool.starmap(generate_video, params)
    for dpath in df['grandparent_dir'].unique():
        generate_video(args.root, dpath)


def generate_readme(args):
    md_path = "list.md"
    csv_path = os.path.splitext(args.out)[0] + "_groups.csv"
    df = pd.read_csv(csv_path)

    # markdown_table = df.to_markdown(index=False)
    markdown_content = ""

    # with open(md_path, "w") as file:
    #     file.write(markdown_table)
    for grandparent_dir in df['grandparent_dir'].unique():  # todo
        # Add the grandparent_dir as a second-level heading
        markdown_content += f"## {grandparent_dir}\n\n"

        # Filter the DataFrame for rows with this grandparent_dir
        filtered_df = df[df['grandparent_dir'] == grandparent_dir]

        mp4_path = "mp4/" + grandparent_dir.replace("/", ".") + ".mp4"
        markdown_content += f"![]({mp4_path})\n"

        # Iterate over the rows in the filtered DataFrame
        for _, row in filtered_df.iterrows():
            # For each row, list the other attributes
            for col in filtered_df.columns:
                if col == 'average_frames':
                    markdown_content += f"- **{col}:** {row[col]:.1f}\n"
                elif col == 'first_time_stamp' or col == 'last_time_stamp':
                    # Parse the string into a datetime object
                    dt = datetime.strptime(row[col], "%Y-%m-%d %H:%M:%S.%f")
                    # Round the seconds
                    # Check if microseconds are greater than or equal to 500000, then add one second
                    if dt.microsecond >= 500000:
                        dt = dt + timedelta(seconds=1)
                    # Set microseconds to 0 as we are rounding to the nearest second
                    dt = dt.replace(microsecond=0)
                    # Convert the datetime object back to a string
                    rounded_datetime_string = dt.strftime("%Y-%m-%d %H:%M:%S")

                    markdown_content += f"- **{col}:** {rounded_datetime_string}\n"
                elif col != 'grandparent_dir':
                    markdown_content += f"- **{col}:** {row[col]}\n"
            markdown_content += "\n"

    # Write the markdown content to the specified markdown file
    with open(md_path, "w") as file:
        file.write(markdown_content)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="root directory of pkls")
    parser.add_argument("out", type=str, help="output path of csv")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="",
    )
    args = parser.parse_args()

    return args


def main():
    random.seed(1)
    args = get_args()
    convert(args)
    generate_sample_videos(args)
    generate_readme(args)


if __name__ == "__main__":
    main()
