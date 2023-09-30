import json
import os

import numpy as np
import torch
from torch.utils.data import Sampler

HMDB51_VIDEO_MAPPING_PATH = "resources/syn4vl_hmdb51_mapping.json"

class ImagenetHandler:
    IMAGENET_DATA_PATH = "datasets/imagenet/ILSVRC/Data/CLS-LOC/train"
    IMAGENET_NAMES_PATH = "datasets/imagenet/ILSVRC/Annotations/CLS-LOC/imagenet_class_index.json"
    IMAGENET_VIDEO_MAPPING_ACTIONS_PATH = "resources/syn4vl_imagenet_actions_mapping.json"
    IMAGENET_VIDEO_MAPPING_OBJECTS_PATH = "resources/syn4vl_imagenet_objects_mapping.json"

    def __init__(self):
        self.imagenet_mapping = {}
        if ImagenetHandler.does_imagenet_exist():
            with open(ImagenetHandler.IMAGENET_VIDEO_MAPPING_ACTIONS_PATH) as f:
                self.imagenet_mapping.update(json.load(f))

            with open(ImagenetHandler.IMAGENET_VIDEO_MAPPING_OBJECTS_PATH) as f:
                self.imagenet_mapping.update(json.load(f))
            
            with open(ImagenetHandler.IMAGENET_NAMES_PATH) as f:
                imagenet_names = json.load(f)
        else:
            print("Could not find imagenet pairings files")

        self.imagenet_names = {v[0]: v[1] for k, v in imagenet_names.items()}
        del imagenet_names

    def select_random(self, frame_path):
        if frame_path in self.imagenet_mapping:
            categories = self.imagenet_mapping[frame_path]
        else:
            return

        cat = np.random.choice(categories)
        cat_path = os.path.join(ImagenetHandler.IMAGENET_DATA_PATH, cat)
        img_name = np.random.choice(os.listdir(cat_path))
        img_path = os.path.join(cat_path, img_name)

        img_txt = self.imagenet_names[cat]
        img_txt = (
            "a photo of "
            + ("an " if img_txt[0] in "aieuo" else "a ")
            + img_txt.replace("_", " ")
        )
        return img_path, img_txt
    
    @staticmethod
    def does_imagenet_exist():
        return (os.path.isfile(ImagenetHandler.IMAGENET_VIDEO_MAPPING_ACTIONS_PATH) and os.path.isfile(ImagenetHandler.IMAGENET_VIDEO_MAPPING_OBJECTS_PATH) and os.path.isfile(ImagenetHandler.IMAGENET_NAMES_PATH))

class IndexRandomSampler(Sampler):
    def __init__(self, dataset, use_real=False):
        # Assumes the first dataset in the concat is the action dataset
        self.use_real = use_real
        if use_real:
            new_indices = []
            for i in np.arange(0, len(dataset), 2):
                new_indices.append([i, i + 1])
        else:
            new_indices = np.arange(len(dataset))

        self.indices = new_indices

    def __iter__(self):
        for i in torch.randperm(len(self.indices)):
            if isinstance(self.indices[i], list):
                yield from iter(self.indices[i])
            else:
                yield self.indices[i]

    def __len__(self):
        return 2 * len(self.indices) if self.use_real else len(self.indices)
