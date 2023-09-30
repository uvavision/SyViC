import json
import os
import random

from PIL import Image, ImageOps
from tdw.librarian import ModelLibrarian
from torch.utils.data import Dataset


class descriptors:
    BIG = [
        "large",
        "big",
        "tall",
        "long",
        "huge",
        "full",
        "heavy",
        "high",
        "wide",
        "thick",
        "huge",
        "giant",
        "skinny",
        "full",
        "fat",
        "bigger",
        "taller",
        "longer",
        "heavier",
        "higher",
        "wider",
        "thicker",
    ]

    SMALL = [
        "small",
        "short",
        "little",
        "thin",
        "light",
        "narrow",
        "low",
        "tiny",
        "smaller",
        "shorter",
        "thinner",
        "lighter",
        "narrower",
        "lower",
    ]

    COMODIN = [
        "large tree",
        "large building",
        "large rock",
        "large window",
        "long neck",
        "large elephant",
        "long hair",
        "large ear",
        "tall building",
        "large bus",
        "large umbrella",
        "long tail",
        "large mirror",
        "tall tree",
        "large windows",
        "large clock",
        "large sign",
    ]


class TDW_Size(Dataset):
    def __init__(
        self,
        data_path="datasets/syvic/Size",
        transform=None,
        sample_fps=30,
        load_filepaths=False,
        use_real=False,
        imagenet_handler=None,
    ):
        self.transform = transform
        self.load_filepaths = load_filepaths
        use_real = use_real and (imagenet_handler is not None)

        captions_path = os.path.join(data_path, "captions.json")
        with open(captions_path) as f:
            captions = json.load(f)

        file_list = []
        text_list = []
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

        library = ModelLibrarian(library="models_full.json")
        self.all_tdw_records = {}
        for rec in library.records:
            self.all_tdw_records[rec.name] = rec.wcategory

        self.big = descriptors.BIG
        self.small = descriptors.SMALL
        self.comodin = descriptors.COMODIN

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        text = text.split("*-*")

        if len(text) == 1:
            # from Imagenet
            text = text[0]
        else:
            obj_right = self.all_tdw_records[text[1]]
            obj_left = self.all_tdw_records[text[2]]

            desc1 = f"a {random.choice(self.small)} {obj_right} and a {random.choice(self.big)} {obj_left}"
            desc2 = f"{random.choice(self.big)} {obj_left}"
            desc3 = f"{random.choice(self.small)} {obj_right}"
            desc4 = random.choice(self.comodin)
            text = random.choice([desc1, desc2, desc3, desc4])

        if self.load_filepaths:
            return self.file_list[idx], text

        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")

        should_flip = random.choice([True, False])
        if should_flip:
            img = ImageOps.mirror(img)

        if self.transform:
            img = self.transform(img)

        return img, text
