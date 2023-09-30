import sys
sys.path.append("../lib/")

import json
import os
import sys

import clip as orig_clip
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from src.lib.CLIP.clip import *

class VLChecklist_DATA(Dataset):
    def __init__(
        self,
        json_data,
        transform=None,
        root_path="~/datasets/data/VG/images/VG_100K_ALL/{}",
    ):
        self.file_list = []
        self.positive_text_list = []
        self.negative_text_list = []

        for d in json_data:
            file_path = root_path.format(d[0]).replace("VG_100K_2", "VG_100K")
            if not os.path.isfile(file_path):
                continue
            self.file_list.append(file_path)
            self.positive_text_list.append(d[1]["POS"][0])
            self.negative_text_list.append(d[1]["NEG"][0])
        print(f"Collected {len(self.file_list)} images for {root_path.format('')}")

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.positive_text_list[idx],
            self.negative_text_list[idx],
            self.file_list[idx],
        )


class VLChecklist:
    def __init__(
        self,
        gentype_="Relation",
        fromset_="vg",
        dtype_="spatial",
        root_path="~/datasets/data/VG/images/VG_100K_ALL/{}",
    ):
        attribute_path = f"~/datasets/{gentype_}/{fromset_}/{dtype_}.json"
        if not os.path.isfile(attribute_path):
            attribute_path = f"~/datasets/{gentype_}/{fromset_}_{dtype_}.json"
        with open(attribute_path) as f:
            data = json.load(f)
        self.data = data
        self.root_path = root_path

    def eval(self, model, preprocess):
        checklist_dataset = VLChecklist_DATA(
            self.data, transform=preprocess, root_path=self.root_path
        )
        dataset_loader = torch.utils.data.DataLoader(
            checklist_dataset, batch_size=400, shuffle=False, num_workers=8
        )

        device = "cuda"

        acc = 0
        total = 0
        for images, pos_samples, neg_samples, img_paths in tqdm(dataset_loader):
            images = images.to(device)
            pos_text = orig_clip.tokenize(pos_samples).to(device)
            neg_text = orig_clip.tokenize(neg_samples).to(device)

            with torch.no_grad():
                image_features = model.encode_image(images)
                pos_text_features = model.encode_text(pos_text)
                neg_text_features = model.encode_text(neg_text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            pos_text_features /= pos_text_features.norm(dim=-1, keepdim=True)
            neg_text_features /= neg_text_features.norm(dim=-1, keepdim=True)

            per_sample_res = [
                torch.tensor(
                    [
                        pos_text_features[i].cpu().numpy()
                        @ image_features[i].cpu().numpy().T,
                        neg_text_features[i].cpu().numpy()
                        @ image_features[i].cpu().numpy().T,
                    ]
                )
                .float()
                .softmax(dim=-1)
                for i in range(len(images))
            ]
            for s_res in per_sample_res:
                total += 1
                if s_res[0] > s_res[1]:
                    acc += 1

        print(acc, total, acc / total)
        return acc / total, 0

    def get_dataset(self, preprocess=None):
        return VLChecklist_DATA(
            self.data, transform=preprocess, root_path=self.root_path
        )
