import os

import numpy as np
import pickle
import csv

from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging

import utils.prompting_utils as putils

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    filename="logs/invalid_videos.log",
    filemode="a",
)

invalid_factor = 0.003

def identify_invalid_frames(video_idx):
    metadata_path = f"{DATA_DIR}/{video_idx}/info.p"
    if not os.path.isfile(metadata_path):
        return
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    for c in os.listdir(f"{DATA_DIR}/{video_idx}"):
        if not c.startswith("c"):
            continue
        video_path = Path(f"{DATA_DIR}/{video_idx}/{c}").resolve()
        if str(video_path) in LOADED_VIDEOS:
            continue

        frames_validities = {i: [] for i in range(6)}
        for i in metadata:
            if i == 0:
                continue
            idx = "0" * (4 - len(str(i))) + str(i)
            path_to_img = f"{DATA_DIR}/{video_idx}/{c}/category_{idx}.png"
            if not os.path.isfile(path_to_img):
                logging.warning(f"Couldn't find {path_to_img}. Skipping.")
                continue
            human_ids = [
                h
                for h in metadata[i]["humans"]
                if metadata[i]["humans"][h].get("action_description")
            ]
            hid = metadata[i]["humans"][human_ids[0]].get("category_id")
            valid_code = putils.is_valid_frame(path_to_img, hid)
            frames_validities[valid_code].append(i)

        with open(CSV_FILE, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    video_path,
                    len(frames_validities[2]),
                    frames_validities[0],
                    frames_validities[1],
                    frames_validities[2],
                    frames_validities[3],
                    frames_validities[4],
                    frames_validities[5],
                ]
            )
        logging.info(
            f"Done with video {video_idx}/{c}. Found {len(frames_validities[0])} humanless frames."
        )


if __name__ == "__main__":
    DATA_DIR = "datasets/Action"
    CSV_FILE = "datasets/syvic/Action/invalid_videos_action.csv"

    LOADED_VIDEOS = set()
    lines = [["video_path", "num_valid", "0", "1", "2", "3", "4", "5"]]
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE) as f:
            for line in csv.reader(f):
                LOADED_VIDEOS.add(line[0])
                lines.append(line)

    # Write CSV header
    with open(CSV_FILE, "w+") as f:
        writer = csv.writer(f)
        writer.writerows(lines)

    # Launch processes for identifying valid/invalid frames
    folders = os.listdir(DATA_DIR)
    nprocs = min(cpu_count(), len(folders))
    with Pool(nprocs) as pool:
        pool.map(identify_invalid_frames, folders)
