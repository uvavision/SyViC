import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .function import adaptive_instance_normalization, coral
from .net import decoder, vgg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HMDB51_VIDEO_MAPPING_PATH = "resources/json/syn4vl_hmdb51_mapping.json"


class AdaIN:
    def __init__(
        self,
        decoder_path="adain/models/decoder.pth",
        vgg_path="adain/models/vgg_normalised.pth",
    ):
        # Load encoder
        self.vgg = vgg
        self.vgg.eval()
        self.vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        self.vgg.to(device)

        # Load decoder
        self.decoder = decoder
        self.decoder.eval()
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.decoder.to(device)

        # Load transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    @torch.no_grad()
    def transfer(self, contents, style, preserve_color=True, alpha=0.7):
        contents = contents.to(device, dtype=torch.float)
        style = style.to(device, dtype=torch.float)
        if preserve_color:
            try:
                style = coral(style, contents[0])
            except Exception as E:
                print(E)
                print("Style shape:", style.shape)
                print("Contents shape:", contents[0].shape)

        content_f = self.vgg(contents)
        style_f = self.vgg(style.unsqueeze(0))
        style_f = style_f.expand(content_f.shape[0], *style_f.shape[1:])
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)

        return self.decoder(feat)

    def transfer_paths(self, content_paths, style_path, preserve_color=True, alpha=0.7):
        contents = []
        for content_path in content_paths:
            content = self.transform(Image.open(str(content_path)))
            contents.append(content)
        contents = torch.stack(contents)
        style = self.transform(Image.open(str(style_path)))
        return self.transfer(contents, style, preserve_color, alpha)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
