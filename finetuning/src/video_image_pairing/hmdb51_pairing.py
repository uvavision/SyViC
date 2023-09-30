import csv
import json
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

hmdb51_path = "datasets/hmdb51"
hmdb51_image_paths = glob(f"{hmdb51_path}/**/**/*.png")

csv_path = "datasets/syvic/Action/invalid_videos_action.csv"
video_images_map_path = "resources/syvic_hmdb51_mapping.json"

model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

hmdb51_embs = []

batch_imgs = []
bz = 256
for i, img_path in tqdm(enumerate(hmdb51_image_paths), total=len(hmdb51_image_paths)):
    img = Image.open(img_path)
    img = preprocess(img)
    batch_imgs.append(img)
    if (i+1) % bz != 0 and i != (len(hmdb51_image_paths) - 1):
        continue
    batch_imgs = torch.stack(batch_imgs).to(device)
    
    with torch.no_grad():
        out = model(batch_imgs)
        hmdb51_embs.append(out.cpu().numpy())
    batch_imgs = []

hmdb51_embs = np.concatenate(hmdb51_embs).squeeze()


videos_images_mapping = {}

video_paths = []
with open(csv_path) as f:
    reader = csv.reader(f)
    lines = list(reader)[1:]
    for l in lines:
        if int(l[1]) > 0:
            video_paths.append(l[0])

video_objects_embeddings = []
for v in tqdm(video_paths):
    trg_img_path = np.random.choice(glob(os.path.join(v, "img*")))
    trg_img = Image.open(trg_img_path)
    trg_img = preprocess(trg_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(trg_img)
        video_objects_embeddings.append(out.cpu().numpy().squeeze())

video_objects_embeddings = np.stack(video_objects_embeddings)

def cosine(u, v):
    return np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v))

cos_sim = cosine(video_objects_embeddings, hmdb51_embs)
cos_sim_argmax = cos_sim.argmax(axis=1)

for i in range(cos_sim_argmax.shape[0]):
    vpath = video_paths[i]
    video_image_paths = hmdb51_image_paths[cos_sim_argmax[i]]
    videos_images_mapping[vpath] = video_image_paths

with open(video_images_map_path, "w+") as f:
    json.dump(videos_images_mapping, f, indent=4)
