import csv
import json
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def cosine(u, v):
    return np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v))


imgnet_path = "datasets/imagenet/ILSVRC/Data/CLS-LOC/train"
names_path = "datasets/imagenet/ILSVRC/Annotations/CLS-LOC/imagenet_class_index.json"

validity_csv_path = "datasets/syvic/Action/invalid_videos_action.csv"
video_images_map_path = "resources/syvic_imagenet_objects_mapping.json"

model = SentenceTransformer("all-mpnet-base-v2")

with open(names_path) as f:
    names = json.load(f)

imagenet_names = []
imagenet_embs = []
for n in tqdm(names):
    txt = names[n][1]
    full_name = names[n][0]
    imagenet_names.append(full_name)
    imagenet_embs.append(model.encode(txt))
imagenet_embs = np.stack(imagenet_embs)

videos_images_mapping = {}
video_paths = []
with open(validity_csv_path) as f:
    reader = csv.reader(f)
    lines = list(reader)[1:]
    for l in lines:
        if int(l[1]) > 0:
            video_paths.append(l[0])

video_objects_embeddings = []
for v in tqdm(video_paths):
    video_c = v.split("/")[-1]
    video_objects = ""
    if video_c.startswith("c"):
        info_path = v.replace("/" + video_c, "/info.p")
    else:
        info_path = os.path.join(v, "info.p")
    if not os.path.isfile(info_path):
        print(info_path)
        continue
    with open(info_path, "rb") as f:
        scene_objects = pickle.load(f)[0]["objects"]
    for obj in scene_objects:
        obj_cat = scene_objects[obj]["category_name"]
        video_objects += obj_cat + ", "
    video_objects = video_objects[:-2]
    video_objects_embeddings.append(model.encode(video_objects))
video_objects_embeddings = np.stack(video_objects_embeddings)

cos_sim = cosine(video_objects_embeddings, imagenet_embs)
cos_sim_argsort = (-cos_sim).argsort(axis=1)[:, :3]

for i in range(cos_sim_argsort.shape[0]):
    vpath = video_paths[i]
    video_image_paths = [imagenet_names[img_idx] for img_idx in cos_sim_argsort[i]]
    videos_images_mapping[vpath] = video_image_paths

with open(video_images_map_path, "w+") as f:
    json.dump(videos_images_mapping, f, indent=4)
