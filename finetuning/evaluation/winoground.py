import sys
sys.path.append("../lib/")

import torch
import os
import numpy as np
from tqdm import tqdm
import clip
from datasets import load_dataset


class WINOGROUND:
    def __init__(self):
        auth_token = os.environ.get("HF_TOKEN")
        winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)[
            "test"
        ]
        self.winoground = winoground

    def text_correct(self, result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(self, result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(self, result):
        return self.image_correct(result) and self.text_correct(result)

    def eval(self, model, preprocess):
        position_idxs = [10, 11, 12, 30, 31, 35, 38, 39, 43, 52, 62, 81, 83, 100, 112, 117, 119, 126, 130, 154, 156, 158, 166, 177, 180, 181, 182, 183, 184, 185, 186, 187, 188, 200, 201, 203, 214, 216, 230, 236, 237, 239, 240, 241, 242, 244, 247, 248, 255, 258, 261, 268, 277, 278, 279, 289, 290, 291, 297, 320, 344, 345, 347, 352, 355, 361, 362, 367, 372, 382, 383, 387, 391, 392, 395]
        vanilla_idxs = [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 29, 30, 32, 33, 34, 35, 37, 39, 43, 45, 47, 48, 50, 51, 52, 53, 54, 56, 57, 59, 60, 64, 66, 67, 71, 79, 80, 85, 87, 89, 90, 91, 92, 94, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 112, 115, 117, 120, 122, 123, 124, 125, 126, 127, 129, 137, 139, 140, 141, 142, 145, 146, 147, 151, 153, 154, 157, 158, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 175, 177, 178, 179, 180, 181, 183, 184, 185, 186, 194, 195, 196, 197, 202, 205, 207, 212, 213, 216, 225, 231, 236, 240, 243, 244, 248, 250, 251, 252, 256, 259, 261, 265, 266, 269, 270, 271, 272, 273, 278, 279, 283, 285, 288, 289, 290, 291, 294, 297, 301, 302, 306, 308, 309, 317, 328, 337, 341, 349, 357, 360, 366, 368, 369, 370, 372, 378, 379, 380, 389, 391, 397]

        model.eval()
        winoground = self.winoground
        winoground_clip_scores = []
        print(" == FULL SET == ")
        for wino_idx in tqdm(range(len(winoground))):
            caption_0 = [
                winoground[wino_idx]["caption_0"],
                winoground[wino_idx]["caption_1"],
            ]
            image_0 = [preprocess(winoground[wino_idx]["image_0"].convert("RGB"))]
            image_input0 = torch.tensor(np.stack(image_0)).cuda()
            text_tokens0 = clip.tokenize(caption_0).cuda()
            with torch.no_grad():
                image_features0 = model.encode_image(image_input0).float()
                text_features0 = model.encode_text(text_tokens0).float()
            image_features0 /= image_features0.norm(dim=-1, keepdim=True)
            text_features0 /= text_features0.norm(dim=-1, keepdim=True)
            similarity0 = text_features0.cpu().numpy() @ image_features0.cpu().numpy().T

            caption_0 = [
                winoground[wino_idx]["caption_0"],
                winoground[wino_idx]["caption_1"],
            ]
            image_0 = [preprocess(winoground[wino_idx]["image_1"].convert("RGB"))]
            image_input0 = torch.tensor(np.stack(image_0)).cuda()
            text_tokens0 = clip.tokenize(caption_0).cuda()
            with torch.no_grad():
                image_features0 = model.encode_image(image_input0).float()
                text_features0 = model.encode_text(text_tokens0).float()
            image_features0 /= image_features0.norm(dim=-1, keepdim=True)
            text_features0 /= text_features0.norm(dim=-1, keepdim=True)
            similarity1 = text_features0.cpu().numpy() @ image_features0.cpu().numpy().T

            # get all results
            winoground_clip_scores.append(
                {
                    "id": winoground[wino_idx]["id"],
                    "c0_i0": similarity0[0],
                    "c0_i1": similarity1[0],
                    "c1_i0": similarity0[1],
                    "c1_i1": similarity1[1],
                }
            )

        # compute full scores
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in winoground_clip_scores:
            text_correct_count += 1 if self.text_correct(result) else 0
            image_correct_count += 1 if self.image_correct(result) else 0
            group_correct_count += 1 if self.group_correct(result) else 0
        denominator = len(winoground_clip_scores)
        print("text score:", text_correct_count / denominator)
        print("image score:", image_correct_count / denominator)
        print("group score:", group_correct_count / denominator)
        full_set_res = {
            "text_score": text_correct_count / denominator,
            "image_score": image_correct_count / denominator,
            "group_score": group_correct_count / denominator,
        }

        # compute position scores
        print(" == POSITION SUBSET == ")
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in tqdm(np.array(winoground_clip_scores)[position_idxs]):
            text_correct_count += 1 if self.text_correct(result) else 0
            image_correct_count += 1 if self.image_correct(result) else 0
            group_correct_count += 1 if self.group_correct(result) else 0
        denominator = len(np.array(winoground_clip_scores)[position_idxs])
        print("text score:", text_correct_count / denominator)
        print("image score:", image_correct_count / denominator)
        print("group score:", group_correct_count / denominator)
        position_set_res = {
            "text_score": text_correct_count / denominator,
            "image_score": image_correct_count / denominator,
            "group_score": group_correct_count / denominator,
        }

        # compute vanilla scores
        print(" == VANILLA SUBSET == ")
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in tqdm(np.array(winoground_clip_scores)[vanilla_idxs]):
            text_correct_count += 1 if self.text_correct(result) else 0
            image_correct_count += 1 if self.image_correct(result) else 0
            group_correct_count += 1 if self.group_correct(result) else 0
        denominator = len(np.array(winoground_clip_scores)[vanilla_idxs])
        print("text score:", text_correct_count / denominator)
        print("image score:", image_correct_count / denominator)
        print("group score:", group_correct_count / denominator)
        vanilla_set_res = {
            "text_score": text_correct_count / denominator,
            "image_score": image_correct_count / denominator,
            "group_score": group_correct_count / denominator,
        }

        return full_set_res, position_set_res, vanilla_set_res
