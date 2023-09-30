import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class TDW_Action(Dataset):
    def __init__(
        self,
        actions_captions_path="datasets/syvic/Action/captions",
        actions_captions_valid_path="datasets/syvic/Action/invalid_videos_action.csv",
        captioning_model="grammar_full",
        sample_type="None",
        sample_fps=5,
        dataset_subset=1,
        transform=None,
        use_real=False,
        load_filepaths=False,
        imagenet_handler=None,
    ):
        """
        Initialize TDW Action dataset.

        :param actions_captions_path: Path to the directory containing action captions.
        :param actions_captions_valid_path: Path to the validation file for captions.
        :param captioning_model: Model to be used for captioning.
        :param sample_type: Type of sampling to be used.
        :param sample_fps: Sampling rate in frames per second.
        :param dataset_subset: Fraction of dataset to be used.
        :param transform: Optional transformations to be applied on images.
        :param use_real: Flag to determine whether to use real images or not.
        :param load_filepaths: Flag to determine whether to load file paths or actual images.
        :param imagenet_handler: A handler to use ImageNet for matching synthetic to real images.
        """

        # Initialize instance variables
        self.use_real = use_real and (imagenet_handler is not None)
        self.sample_fps = sample_fps
        self.sample_type = sample_type
        self.transform = transform
        self.load_filepaths = load_filepaths
        self.captioning_model = captioning_model
        self.dataset_subset = dataset_subset
        self.imagenet_handler = imagenet_handler

        self.video_caption_mapping = self._build_video_to_caption_mapping(
            actions_captions_path
        )
        videos_valid_frames = self._sample_valid_videos(actions_captions_valid_path)
        file_list_all, text_list_all = self._get_image_caption_mapping(
            videos_valid_frames
        )

        indices = self._sample_frames(len(file_list_all))
        self.file_list = [file_list_all[i] for i in indices]
        self.text_list = [text_list_all[i] for i in indices]

        print(
            f"ACTION FRAMES: Retrived {len(file_list_all)} frames, Sampled {len(self.file_list)} frames"
        )

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset given an index.

        :param idx: Index of the item to be retrieved.
        :return: Image and corresponding text.
        """
        if self.load_filepaths:
            return self.file_list[idx], self.text_list[idx]
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        text = self.text_list[idx]
        text = text.replace("first", "").replace("second", "").replace("third", "")
        return img, text

    def _sample_frames(self, n):
        """
        Sample frames based on the provided type.

        :param n: Total number of frames.
        :return: Indices of the sampled frames.
        """
        # Choose frames based on the specified sample_type
        sample_type = self.sample_type
        if sample_type == "random":
            indices = np.random.choice(
                np.arange(n), int(np.ceil(n * self.sample_fps / 30)), replace=False
            )
        elif sample_type == "uniform":
            indices = []
            skip_every = 30 // self.sample_fps
            if self.use_real:
                skip_every *= 2
            for ii in np.arange(0, n, int(skip_every)):
                indices.append(ii)
                if self.use_real:
                    indices.append(ii + 1)
        elif sample_type == "uniform_delayed":
            indices = []
            skip_every = 30 // self.sample_fps
            if self.use_real:
                skip_every *= 2
            for ii in np.arange(20, n, int(skip_every)):
                indices.append(int(ii - 1))
                if self.use_real:
                    indices.append(int(ii + 1))
        else:
            indices = np.arange(n)
        return indices

    def _get_images_and_captions_mapping(self, videos_valid_frames):
        """
        Builds a list of image paths and corresponding textual captions.

        :param videos_valid_frames: A dictionary containing valid frames for each video.
        :return: Lists of file paths and corresponding texts.
        """
        file_list_all = []
        text_list_all = []

        caption_prefix_len = len("In this scene, we can see ")
        video_captions = list(self.video_caption_mapping.items())
        for video_path, captions_path in tqdm(video_captions):
            if video_path not in videos_valid_frames:
                continue
            with open(captions_path) as captions_f:
                captions = json.load(captions_f)

            for frame, frame_captions in captions.items():
                if (
                    (frame not in videos_valid_frames[video_path])
                    or (self.captioning_model not in frame_captions)
                    or len(frame_captions[self.captioning_model]) < caption_prefix_len
                ):
                    continue

                frame_cap = frame_captions[self.captioning_model].strip()
                frame_cap = frame_cap.split(".")
                sampled_sentences = [
                    ci.strip()
                    for ci in frame_cap
                    if (np.random.random() < (7 / len(frame_cap)))
                ]
                frame_cap = ". ".join(sampled_sentences).strip()

                img_idx = f"{int(frame):04d}"
                img_path = os.path.join(video_path, f"img_{img_idx}.jpg")

                if self.use_real:
                    imgnet_corresponding = self.imagenet_handler.select_random(
                        video_path
                    )
                    if imgnet_corresponding:
                        imgnet_path, imgnet_caption = imgnet_corresponding
                        file_list_all.append(imgnet_path)
                        text_list_all.append(imgnet_caption)
                    else:
                        print(f"Couldn't find image for {video_path}")

                file_list_all.append(img_path)
                text_list_all.append(frame_cap)
        return file_list_all, text_list_all

    def _sample_valid_videos(self, validation_path):
        """
        Sample valid videos.

        :param validation_path: Path to the validation CSV file.
        :return: Dictionary mapping videos to their valid frames.
        """
        videos_valid_frames = {}
        with open(validation_path) as f:
            reader = csv.reader(f)
            valid_index = next(reader).index("2")
            for line in reader:
                path = line[0]
                valid_frames = [str(i) for i in eval(line[valid_index])]
                if valid_frames:
                    videos_valid_frames[path] = valid_frames

        # Select sample of the dataset, if specified
        valid_videos = list(videos_valid_frames.keys())
        if self.dataset_subset != 1:
            sampled_videos = np.random.choice(
                valid_videos,
                int(self.dataset_subset * len(valid_videos)),
                replace=False,
            )
            videos_valid_frames = {v: videos_valid_frames[v] for v in sampled_videos}
        print(
            f"ACTION FRAMES: Retrieved {len(valid_videos)} valid videos. Sampled {len(videos_valid_frames)} videos."
        )
        return videos_valid_frames

    def _build_video_to_caption_mapping(self, full_captions_path):
        """
        Build a mapping from videos to their caption files.

        :param full_captions_path: Path to the directory containing full captions.
        :return: Dictionary mapping videos to their caption files.
        """
        video_caption_mapping = {}
        for video_idx in os.listdir(full_captions_path):
            video_path = Path(os.path.join(full_captions_path, video_idx)).resolve()
            for view in os.listdir(video_path):
                if not view.startswith("c"):
                    continue
                view_path = os.path.join(video_path, view)
                captions_path = os.path.join(
                    full_captions_path, f"captions/captions_{video_idx}_{view}.json"
                )
                if not os.path.isfile(captions_path):
                    continue
                video_caption_mapping[str(view_path)] = captions_path
        return video_caption_mapping
