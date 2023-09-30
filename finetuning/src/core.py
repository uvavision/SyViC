import datetime
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import src.utils.data as data_utils
import torch
import torch.nn as nn
from PIL import Image
from src.adain.utils import AdaIN
from src.datasets import dataset_types
from src.lib.CLIP.clip import clip
from src.lib.CLIP.clip.model import convert_weights
from src.random_augmenter import random_augment
from src.utils.augmentations import RandomAugmentation
from torch.utils.data import ConcatDataset
from tqdm import tqdm


class FT_CLIP:
    """
    Class for fine-tuning CLIP with specified configurations.
    """

    def __init__(self, args):
        """
        Initialize the FT_CLIP object with provided arguments.
        :param args: Object containing training parameters.
        """
        self.args = args
        if data_utils.ImagenetHandler.does_imagenet_exist():
            self.imagenet_handler = data_utils.ImagenetHandler()
        else:
            self.imagenet_handler = None

        print("- set seed:", args.seed)
        self.seed_everything(args.seed)

        # Load model and freeze visual head if needed
        print("- load model")
        self.load_model(base_name=args.backbone_name, weight_name=args.load_from_path)
        if self.args.frozen:
            print("- freeze visual head")
            self.freeze_visual_head()

        # Setting to training mode
        print("- set train model")
        self.set_train_state()

        # Load style module
        self.load_style_module()

        # Load train dataset
        print(f"- load dataset - {args.data_type} ")
        self.load_train_dataset(args.data_type, args.heavy_aug)

        # Train the model
        print("- start training")
        self.train_model()

        # Done!
        print("done...")

    def seed_everything(self, seed: int):
        """
        Set a random seed for reproducibility.
        :param seed: Integer seed value.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def load_model(self, base_name="RN50", weight_name=""):
        """
        Load the CLIP model and display its parameters.

        :param base_name: Base model name.
        :param weight_name: Pre-trained weight path.
        """

        self.model, self.preprocess = clip.load(
            base_name,
            jit=False,
            lora=self.args.lora_r,
            mixstyle=self.args.mixstyle,
        )
        self.model = self.model.cuda()
        if self.args.lora_r <= 0:
            convert_weights(self.model)

        if weight_name:
            self.model.load_state_dict(torch.load(weight_name)["model"])

        print("=========")
        print(
            "Model parameters:",
            f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}",
        )
        print("Input resolution:", self.model.visual.input_resolution)
        print("Context length:", self.model.context_length)
        print("Vocab size:", self.model.vocab_size)
        print("=========")

    def load_style_module(self):
        """
        Load style module and HMDB51 video mapping paths.
        """
        if self.args.style_transfer:
            self.style_module = AdaIN()
            with open(data_utils.HMDB51_VIDEO_MAPPING_PATH) as f:
                self.hmdb51_paths = list(json.load(f).values())
        else:
            self.style_module = None

    def set_train_state(self):
        self.model.cuda().train()

    def set_eval_state(self):
        self.model.cuda().eval()

    def convert_models_to_fp32(self, model):
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
                p.grad.data = p.grad.data.float()

    def load_train_dataset(self, data_type, heavy_aug):
        """
        Load training dataset with optional augmentations.

        :param data_type: Type of data to be loaded.
        :param heavy_aug: Boolean flag to use heavy augmentations.
        """
        if heavy_aug:
            self.preprocess.transforms.insert(
                0, RandomAugmentation(random_augment())
            )  # adding random_augmentations

        ##############################
        dict_all_dataset_types = {}
        type_data = data_type.split(",")

        for subset in {"position", "color", "material", "size"}:
            if subset in type_data:
                dict_all_dataset_types[subset] = dataset_types[subset](
                    transform=self.preprocess,
                    sample_fps=self.args.sample_fps,
                    use_real=self.args.mixstyle,
                    imagenet_handler=self.imagenet_handler,
                )
        if "action" in type_data:
            dict_all_dataset_types["action"] = dataset_types["action"](
                sample_type="uniform_delayed",
                transform=self.preprocess,
                num_videos=self.args.num_videos,
                captioning_model=self.args.action_captions,
                sample_fps=self.args.sample_fps,
                use_real=self.args.mixstyle,
                style_transfer=self.args.style_transfer,
                imagenet_handler=self.imagenet_handler,
            )

        self.tdw_dataset = ConcatDataset([dict_all_dataset_types[t] for t in type_data])
        self.dataset_loader = torch.utils.data.DataLoader(
            self.tdw_dataset,
            batch_size=self.args.batch_size,
            sampler=data_utils.IndexRandomSampler(
                self.tdw_dataset, use_real=self.args.mixstyle
            ),
            num_workers=self.args.workers,
            drop_last=True,
        )

    def freeze_visual_head(self):
        """
        Freeze the visual encoder layers of the model.
        """
        for name, param in self.model.named_parameters():
            if "visual" in name:
                param.requires_grad = False

    def construct_save_path(self, starting_time, extension, suffix=""):
        frozen = "FROZENVisualHead" if self.args.frozen else "ALL"
        arch_name = self.args.backbone_name.replace("/", "")
        folder_name = self.args.folder_name
        if not folder_name:
            folder_name = arch_name

        components = [
            f"{self.args.beta1}B1{self.args.beta2}B2",
            f"{self.args.eps}EPS",
            f"{self.args.weight_decay}WD",
            f"samplefps{self.args.sample_fps}",
            f"LoraRank{self.args.lora_r}",
            f"{starting_time}",
            f"clip_{arch_name}_finetune",
            f"{frozen}",
            f"lr{self.args.lr}",
            f"bs{self.args.batch_size}",
            f"{self.args.steps}epochs",
            f"seed{self.args.seed}",
        ]
        if suffix:
            components.append(suffix)

        fname = "_".join(components) + f".{extension}"
        fpath = f"{folder_name}/{self.args.data_type}/{fname}"

        os.makedirs(f"{folder_name}/{self.args.data_type}", exist_ok=True)
        return fpath.replace(" ", "_").replace(":", "_")

    def train_model(self):
        steps = self.args.steps

        starting_time = datetime.datetime.now()
        print("== Training started at:", starting_time)

        all_losses = []
        all_lrs = []

        self.model.train()
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        model_optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, steps)

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        print("Dataloader len:", len(self.dataset_loader))
        for step in range(steps):
            if step % self.args.save_every == 0 and step > 0:
                if step >= self.args.early_stop:
                    save_path = self.construct_save_path(
                        starting_time, extension="ckpt"
                    )
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "epoch": step,
                            "lr": self.args.lr,
                            "betas": (self.args.beta1, self.args.beta2),
                            "eps": self.args.eps,
                            "weight_decay": self.args.weight_decay,
                        },
                        save_path,
                    )
                    break

            self.model.train()
            log_total_loss = 0
            pbar = tqdm(enumerate(self.dataset_loader), total=len(self.dataset_loader))
            for i, data in pbar:
                images = data[0]
                if self.style_module:
                    with torch.no_grad():
                        trg_img = self.style_module.transform(
                            Image.open(np.random.choice(self.hmdb51_paths))
                        ).cuda()
                        images = self.style_module.transfer(images, trg_img)
                descriptions_per_sample = data[1]
                if self.args.capsplit:
                    split_descriptions_per_sample = []
                    for d in descriptions_per_sample:
                        split_d = [di.strip() for di in d.split(".") if di.strip()]
                        split_d = [
                            ". ".join(split_d[di : di + 4]).strip()
                            for di in range(0, len(split_d), 4)
                        ]
                        split_descriptions_per_sample.append(split_d[:1])
                    descriptions_per_sample = split_descriptions_per_sample

                model_optimizer.zero_grad()

                images = images.cuda()
                descriptions_tokens_atts = []
                for di in descriptions_per_sample:
                    d_tokens = clip.tokenize(di, truncate=True).cuda()
                    descriptions_tokens_atts.append(d_tokens)
                descriptions_tokens_atts = torch.nn.utils.rnn.pad_sequence(
                    descriptions_tokens_atts, batch_first=True
                )

                logits_per_image, logits_per_text = self.model(
                    images, descriptions_tokens_atts
                )
                ground_truth = torch.arange(len(images), dtype=torch.long).cuda()
                total_loss = (
                    loss_img(logits_per_image, ground_truth)
                    + loss_txt(logits_per_text, ground_truth)
                ) / 2

                log_total_loss += total_loss.item()
                model_optimizer.zero_grad()
                total_loss.backward()

                if self.args.lora_r <= 0:
                    self.convert_models_to_fp32(self.model)
                model_optimizer.step()
                if self.args.lora_r <= 0:
                    convert_weights(self.model)
                pbar.set_description(
                    f"epoch {step} batch: {i} - avg. loss: {log_total_loss / (i + 1)}"
                )

            print(
                "epoch: ",
                step,
                " - avg. loss: ",
                log_total_loss / (i + 1),
                " - lr: ",
                model_optimizer.param_groups[0]["lr"],
            )
            all_losses.append(log_total_loss / i)
            all_lrs.append(scheduler.get_last_lr())

            for plt_vals, suffix in zip([all_losses, all_lrs], ["loss", "LRs"]):
                save_path = save_path = self.construct_save_path(
                    starting_time, suffix=suffix, extension="png"
                )
                plt.plot(range(len(plt_vals)), plt_vals)
                plt.savefig(save_path)
                plt.clf()
                plt.cla()
                plt.close()

            scheduler.step()
