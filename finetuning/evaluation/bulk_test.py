import sys

sys.path.append("../")

import argparse
import os
import csv

import torch
import numpy as np
import clip as orig_clip

import winoground
import vl_checklist
from src.lib.CLIP.clip import *

parser = argparse.ArgumentParser(description="FT CLIP MODEL WITH SYNTHETIC")

parser.add_argument(
    "--load_from_path",
    default="",
    type=str,
    help="filename to load from directory path",
)
parser.add_argument(
    "--is_lora", default=False, action="store_true", help="Is using lora"
)
parser.add_argument(
    "--current_arch",
    default="RN50x64",
    choices=["RN50", "RN50x64", "ViT-B32", "ViT-B16"],
    help="clip visual backbone: RN50, RN50x64, ViT-B/32, ViT-B/16",
)
parser.add_argument(
    "--run_baseline",
    default=False,
    action="store_true",
    help="run all tests using baseline model",
)
parser.add_argument(
    "--run_wise_ft",
    default=False,
    action="store_true",
    help="run all WISE-FT using baseline model and model path",
)


def set_model(model_type="ViT-B/32", path="", lora=True, for_wise_ft=False, alpha=1):
    rank = -1
    if lora:
        if path[path.index("LoraRank") + 9] != "_":
            rank = int(path[path.index("LoraRank") + 8 : path.index("LoraRank") + 10])
        else:
            rank = int(path[path.index("LoraRank") + 8])

    print("IS LORA:", lora, "RANK:", rank)

    if lora:
        model, preprocess = clip.load(model_type, jit=False, lora=rank)
    else:
        model, preprocess = orig_clip.load(model_type, jit=False)

    if for_wise_ft:
        vanilla_model_ckpt = model.state_dict()
        ft_model_ckpt = torch.load(path)["model"]
        new_state_dict = interpolate_models(ft_model_ckpt, vanilla_model_ckpt, alpha)
        model.load_state_dict(new_state_dict)
    else:
        if not args.run_baseline:
            model_state = torch.load(path)
            if "model" in model_state.keys():
                model.load_state_dict(model_state["model"])
            else:
                model.load_state_dict(model_state)

    model = model.cuda()
    return model, preprocess


def interpolate_models(theta_0, theta_1, alpha):
    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()
    }
    return theta


if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()

    current_arch = args.current_arch
    print("PATH:", args.load_from_path, current_arch)
    destined_folder = current_arch
    save_results_file = "/".join(args.load_from_path.split("/")[:-1])
    if save_results_file == "":
        save_results_file = ".."

    # test
    wino_eval = winoground.WINOGROUND()
    hake_root_path = "/u/shehadak/storage/datasets/hake/{}"
    swig_root_path = "/u/shehadak/storage/datasets/swig/{}"

    vl_rel_action_hake_eval = vl_checklist.VLChecklist(
        gentype_="Relation", fromset_="hake", dtype_="action", root_path=hake_root_path
    )
    vl_rel_action_swig_eval = vl_checklist.VLChecklist(
        gentype_="Relation", fromset_="swig", dtype_="action", root_path=swig_root_path
    )
    vl_rel_spatial_vg_eval = vl_checklist.VLChecklist(
        gentype_="Relation", fromset_="vg", dtype_="spatial"
    )
    vl_rel_action_vg_eval = vl_checklist.VLChecklist(
        gentype_="Relation", fromset_="vg", dtype_="action"
    )

    vl_att_color_vg_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vg", dtype_="color"
    )
    vl_att_size_vg_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vg", dtype_="size"
    )
    vl_att_material_vg_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vg", dtype_="material"
    )
    vl_att_action_vg_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vg", dtype_="action"
    )
    vl_att_state_vg_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vg", dtype_="state"
    )

    vl_att_color_vaw_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vaw", dtype_="color"
    )
    vl_att_size_vaw_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vaw", dtype_="size"
    )
    vl_att_material_vaw_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vaw", dtype_="material"
    )
    vl_att_action_vaw_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vaw", dtype_="action"
    )
    vl_att_state_vaw_eval = vl_checklist.VLChecklist(
        gentype_="Attribute", fromset_="vaw", dtype_="state"
    )

    if current_arch == "ViT-B32":
        current_arch = "ViT-B/32"
    if current_arch == "ViT-B16":
        current_arch = "ViT-B/16"

    if args.run_wise_ft:
        alpha_vals = [0.10, 0.25, 0.5, 0.75, 0.9]
        print("RUN WISE-FT!!")
    else:
        alpha_vals = [1]

    header = [
        "arch",
        "alpha",
        "ckpt",
        "wino_full_text",
        "wino_full_image",
        "wino_full_group",
        "wino_pos_text",
        "wino_pos_image",
        "wino_pos_group",
        "wino_van_text",
        "wino_van_image",
        "wino_van_group",
        "hake_action",
        "swig_action",
        "vg_spatial",
        "vg_action",
        "vg_color",
        "vg_size",
        "vg_material",
        "vg_action",
        "vg_state",
        "vaw_color",
        "vaw_size",
        "vaw_material",
        "vaw_action",
        "vaw_state",
        "vl-relation",
        "vl-attribute",
        "vl-average",
    ]

    res_file = (
        "/u/shehadak/storage/ckpts/clip/baseline.csv"
        if args.run_baseline
        else f"{save_results_file}/results.csv"
    )
    if not os.path.isfile(res_file):
        with open(res_file, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for alpha in alpha_vals:
        model, preprocess = set_model(
            model_type=current_arch,
            path=args.load_from_path,
            lora=args.is_lora,
            for_wise_ft=args.run_wise_ft,
            alpha=alpha,
        )

        print("RUN :: eval Winoground")
        wino_set_res1, wino_set_res2, wino_set_res3 = wino_eval.eval(model, preprocess)

        print("RUN :: eval VL Hake action")
        vl_set_res1, _ = vl_rel_action_hake_eval.eval(model, preprocess)
        print("RUN :: eval VL Swig action")
        vl_set_res2, _ = vl_rel_action_swig_eval.eval(model, preprocess)
        print("RUN :: eval VL VG spatial, action, color, size, material")
        vl_set_res3, _ = vl_rel_spatial_vg_eval.eval(model, preprocess)
        vl_set_res4, _ = vl_rel_action_vg_eval.eval(model, preprocess)
        vl_set_res5, _ = vl_att_color_vg_eval.eval(model, preprocess)
        vl_set_res6, _ = vl_att_size_vg_eval.eval(model, preprocess)
        vl_set_res7, _ = vl_att_material_vg_eval.eval(model, preprocess)
        vl_set_res8, _ = vl_att_action_vg_eval.eval(model, preprocess)
        vl_set_res9, _ = vl_att_state_vg_eval.eval(model, preprocess)
        print("RUN :: eval VL VAW color, size, material")
        vl_set_res10, _ = vl_att_color_vaw_eval.eval(model, preprocess)
        vl_set_res11, _ = vl_att_size_vaw_eval.eval(model, preprocess)
        vl_set_res12, _ = vl_att_material_vaw_eval.eval(model, preprocess)
        vl_set_res13, _ = vl_att_action_vaw_eval.eval(model, preprocess)
        vl_set_res14, _ = vl_att_state_vaw_eval.eval(model, preprocess)

        vl_rel = np.mean(
            [
                np.mean([vl_set_res1, vl_set_res2, vl_set_res4]),  # Action
                vl_set_res3,  # Spatial
            ]
        )
        vl_att = np.mean(
            [
                np.mean([vl_set_res5, vl_set_res10]),  # Color
                np.mean([vl_set_res6, vl_set_res11]),  # Size
                np.mean([vl_set_res7, vl_set_res12]),  # Material
                np.mean([vl_set_res8, vl_set_res13]),  # Action
                np.mean([vl_set_res9, vl_set_res14]),  # State
            ]
        )
        vl_avg = np.mean([vl_rel, vl_att])

        print("RUN :: WRITE RES TO FILE:")
        print(res_file)
        if args.run_baseline:
            with open(res_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        current_arch,
                        "Baseline",
                        "",
                        wino_set_res1["text_score"],
                        wino_set_res1["image_score"],
                        wino_set_res1["group_score"],
                        wino_set_res2["text_score"],
                        wino_set_res2["image_score"],
                        wino_set_res2["group_score"],
                        wino_set_res3["text_score"],
                        wino_set_res3["image_score"],
                        wino_set_res3["group_score"],
                        vl_set_res1,
                        vl_set_res2,
                        vl_set_res3,
                        vl_set_res4,
                        vl_set_res5,
                        vl_set_res6,
                        vl_set_res7,
                        vl_set_res8,
                        vl_set_res9,
                        vl_set_res10,
                        vl_set_res11,
                        vl_set_res12,
                        vl_set_res10,
                        vl_set_res14,
                        vl_rel,
                        vl_att,
                        vl_avg,
                    ]
                )
        else:
            with open(res_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        current_arch.replace("/", ""),
                        alpha,
                        args.load_from_path,
                        wino_set_res1["text_score"],
                        wino_set_res1["image_score"],
                        wino_set_res1["group_score"],
                        wino_set_res2["text_score"],
                        wino_set_res2["image_score"],
                        wino_set_res2["group_score"],
                        wino_set_res3["text_score"],
                        wino_set_res3["image_score"],
                        wino_set_res3["group_score"],
                        vl_set_res1,
                        vl_set_res2,
                        vl_set_res3,
                        vl_set_res4,
                        vl_set_res5,
                        vl_set_res6,
                        vl_set_res7,
                        vl_set_res8,
                        vl_set_res9,
                        vl_set_res10,
                        vl_set_res11,
                        vl_set_res12,
                        vl_set_res10,
                        vl_set_res14,
                        vl_rel,
                        vl_att,
                        vl_avg,
                    ]
                )
