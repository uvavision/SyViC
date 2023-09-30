import json
import os
from typing import List

import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

import src.adain.utils as adain_utils
from src.utils.data import HMDB51_VIDEO_MAPPING_PATH

BATCH_SIZE = 512
SAVE_DIR = "datasets/syvic_adain/"
IMG_TO_ADAIN_MAPPING_PATH = "resources/syvic_img_to_adain_mapping.json"
IMG_CAPTION_MAPPING_PATH = "resources/syvic_img_caption_mapping.json"
os.makedirs(SAVE_DIR, exist_ok=True)

image_counter = 0
orig_to_adain_mapping = {}


def handle_batch(adain, batch_paths: List[str]):
    global image_counter, orig_to_adain_mapping
    style_path = np.random.choice(hmdb51_paths)
    out = adain.transfer_paths(batch_paths, style_path, alpha=0.25)
    for i in range(len(batch_paths)):
        extension = batch_paths[i].split(".")[-1]
        out_path = os.path.join(SAVE_DIR, "%06d.%s" % (image_counter, extension))
        image_counter += 1

        im = out[i].cpu()
        save_image(im, out_path)
        orig_to_adain_mapping[batch_paths[i]] = out_path


def main(seed: int):
    global orig_to_adain_mapping
    adain_utils.seed_everything(seed)

    content_paths = img_caption_mapping.keys()

    adain = adain_utils.AdaIN()
    batch_paths = []
    for i, content_path in tqdm(enumerate(content_paths), total=len(content_paths)):
        if (i + 1) % BATCH_SIZE == 0:
            handle_batch(adain, batch_paths)
            batch_paths = []
        batch_paths.append(content_path)
    if batch_paths:
        handle_batch(adain, batch_paths)

    with open(IMG_TO_ADAIN_MAPPING_PATH, "w+") as f:
        json.dump(orig_to_adain_mapping, f)
    print("Done converting images. New images saved at path:", SAVE_DIR)
    print("Mapping saved at:", IMG_TO_ADAIN_MAPPING_PATH)


if __name__ == "__main__":
    with open(IMG_CAPTION_MAPPING_PATH) as f:
        img_caption_mapping = json.load(f)

    with open(HMDB51_VIDEO_MAPPING_PATH) as f:
        hmdb51_paths = list(json.load(f).values())

    main(seed=34)
