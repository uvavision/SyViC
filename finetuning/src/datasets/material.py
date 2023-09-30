import json
import os
import random

from PIL import Image, ImageOps
from torch.utils.data import Dataset


class TDW_Material(Dataset):
    def __init__(
        self,
        data_path="datasets/syvic/Material",
        transform=None,
        sample_fps=30,
        load_filepaths=False,
        use_real=False,
        imagenet_handler=None,
    ):
        self.transform = transform
        self.load_filepaths = load_filepaths
        use_real = use_real and (imagenet_handler is not None)

        file_list = []
        text_list = []
        captions_path = os.path.join(data_path, "captions.json")
        with open(captions_path) as f:
            captions = json.load(f)

        for img_id, cap in captions.items():
            img_path = os.path.join(data_path, f"img_{img_id}.jpg")
            if use_real:
                imgnet_corresponding = imagenet_handler.select_random(img_path)
                if imgnet_corresponding:
                    imgnet_path, imgnet_caption = imgnet_corresponding
                    file_list.append(imgnet_path)
                    text_list.append(imgnet_caption)

            file_list.append(img_path)
            text_list.append(cap)

        skip_every = 30 // sample_fps
        if use_real:
            skip_every *= 2

        self.file_list = []
        self.text_list = []
        for i in range(0, len(file_list), int(skip_every)):
            self.file_list.append(file_list[i])
            self.text_list.append(text_list[i])
            if use_real:
                self.file_list.append(file_list[i + 1])
                self.text_list.append(text_list[i + 1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.load_filepaths:
            return self.file_list[idx], self.text_list[idx]

        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if random.choice([True, False]):
            img = ImageOps.mirror(img)

        if self.transform:
            img = self.transform(img)
        text = self.text_list[idx]

        return img, text
