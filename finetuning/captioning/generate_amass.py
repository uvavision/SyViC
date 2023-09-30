import json
import logging
import os
import pickle
import sys

import numpy as np
import torch.multiprocessing as mp
from utils.llm_utils import HuggingfaceHandler
import utils.prompting_utils as putils
import utils.viz_utils as viz
from PIL import Image
from tqdm import tqdm

DATA_PATH = "datasets/syvic/Action"
BZ = 16


def handle_batch(batch, llm_handler):
    batch_prompts, batch_captions = zip(*batch)
    for checkpoint in llm_handler.checkpoints:
        model_caption = llm_handler.handle_request(
            batch_prompts, checkpoint, prefix="In this scene, we can see"
        )
        if model_caption:
            for i, cap in enumerate(model_caption):
                batch_captions[i][checkpoint] = cap


def generate_caption(
    video_idx: int,
    llm_handler: HuggingfaceHandler,
    visualize: bool = False,
    regenerate: bool = False,
):
    logging.info(f"Started captioning video {video_idx}")
    folder_path = os.path.join(DATA_PATH, str(video_idx))
    metadata_path = os.path.join(folder_path, "info.p")

    if not os.path.isfile(metadata_path):
        logging.warning("[Skipping] No file found at: " + metadata_path)
        return
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    pbar = tqdm(total=len(metadata[1]["camera_pos"]) * len(metadata))
    for c in metadata[1]["camera_pos"]:
        captions_path = os.path.join(
            DATA_PATH, "captions_del", f"captions_{video_idx}_{c}.json"
        )
        video_captions = {}
        if os.path.isfile(captions_path):
            with open(captions_path, "rb") as caption_f:
                video_captions = json.load(caption_f)

        batch = []
        for frame_idx in range(1, len(metadata)):
            sample_frame = metadata[frame_idx]
            scene_name = sample_frame.get("tdw_scene_name")

            # Camera position and looking direction
            frame_models = ["grammar", "bloomz", "flan-t5", "grammar_full"]
            if frame_idx in video_captions or str(frame_idx) in video_captions:
                idx_key = frame_idx if frame_idx in video_captions else str(frame_idx)
                frame_models = [
                    m for m in frame_models if m not in video_captions[idx_key]
                ]
            if not frame_models or "grammar" in frame_models:
                continue

            if (
                "grammar" in frame_models
                or "grammar_full" in frame_models
                or regenerate
            ):
                padded_idx = "0" * (4 - len(str(frame_idx))) + str(frame_idx)
                human_ids = {
                    h
                    for h in metadata[frame_idx]["humans"]
                    if metadata[frame_idx]["humans"][h].get("action_description")
                }
                category_img_path = os.path.join(
                    folder_path, c, f"category_{padded_idx}.png"
                )
                if not os.path.isfile(category_img_path):
                    continue
                category_image = np.array(Image.open(category_img_path))
                if not all(
                    [
                        putils.is_valid_frame(
                            category_image,
                            metadata[frame_idx]["humans"][h].get("category_id"),
                        )
                        == 2
                        for h in human_ids
                    ]
                ):
                    continue
                camera_pos = putils.numpify_vector(sample_frame["camera_pos"][c])
                camera_look_at = putils.numpify_vector(sample_frame["camera_look_at"])
                view_matrix = putils.get_view_matrix(camera_pos, camera_look_at)

                objects = []
                for obj_id, object in sample_frame["objects"].items():
                    object_id = putils.numpify_vector(object["category_id"])
                    if not putils.is_obj_in_img(category_image, object_id):
                        continue

                    scene_object = putils.SceneObject.parse(obj_id, object)
                    scene_object.position = putils.transform_to_camera_coordinates(
                        view_matrix, scene_object.position
                    )
                    objects.append(scene_object)

                humans = []
                for human_id, human in sample_frame["humans"].items():
                    human["category_name"] = putils.ordinal(len(humans) + 1) + " human"
                    scene_human = putils.SceneHuman.parse(human_id, human)
                    scene_human.position = putils.transform_to_camera_coordinates(
                        view_matrix, scene_human.position
                    )
                    if scene_human.action:
                        humans.append(scene_human)
                if len(humans) == 1:
                    humans[0].obj_name = "human"

            # Generating captions
            frame_captions = video_captions.setdefault(str(frame_idx), {})
            if "grammar_full" in frame_models or regenerate:
                prompt_full = putils.sample_prompt(
                    [*humans, *objects], category_image, scene_name=scene_name, p=1.0
                )
                frame_captions["grammar_full"] = (
                    prompt_full.replace(putils.PROMPT_PREFIX, putils.STATEMENT_PREFIX)
                    .replace(putils.PROMPT_SUFFIX, "")
                    .strip()
                )

            if "grammar" in frame_models or regenerate:
                prompt = putils.sample_prompt(
                    [*humans, *objects], category_image, scene_name=scene_name
                )
                frame_captions["grammar"] = (
                    prompt.replace(putils.PROMPT_PREFIX, putils.STATEMENT_PREFIX)
                    .replace(putils.PROMPT_SUFFIX, "")
                    .strip()
                )
            prompt = (
                frame_captions["grammar"].replace(
                    "This scene contains", "Please describe a scene containing"
                )
                + " In this scene, we can see"
            )

            batch.append((prompt, frame_captions))
            if len(batch) >= BZ:
                handle_batch(batch, llm_handler)
                batch = []

            if visualize:
                if not os.path.isdir(f"viz/{video_idx}"):
                    os.makedirs(f"viz/{video_idx}")
                viz.viz_frame(
                    frame_idx,
                    [
                        prompt,
                        "... " + frame_captions.get("bloom", ""),
                        "... " + frame_captions.get("bloomz", ""),
                        "... " + frame_captions.get("flan-t5", ""),
                    ],
                    c=c,
                    src_path=folder_path,
                    fontsize=32,
                    outpath=f"viz/{video_idx}/{frame_idx}.png",
                )
                return

            pbar.update(1)
        if batch:
            handle_batch(batch, llm_handler)

        with open(captions_path, "w+") as caption_f:
            json.dump(video_captions, caption_f, indent=2)

    logging.info("Saved captions at " + captions_path)
    return True


def generate_captions_wrapper(**kwargs):
    try:
        generate_caption(**kwargs)
    except Exception as E:
        logging.exception("Raised exception at: " + str(kwargs))


if __name__ == "__main__":
    run_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        filename=f"logs/generate_{run_idx}.log",
        filemode="a",
    )

    num_videos = len(os.listdir(DATA_PATH))
    huggingface_handler = HuggingfaceHandler(["bloomz", "flan-t5"])
    logging.info("Loaded Huggingface Models")

    split_count = num_videos // 20
    for i in range(run_idx * split_count, (run_idx + 1) * split_count):
        generate_captions_wrapper(video_idx=i, llm_handler=huggingface_handler)
