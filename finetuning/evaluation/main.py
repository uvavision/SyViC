"""
You can use this file as a starting point to train your models.
"""
import sys
sys.path.append("../")

import argparse
import glob
import pickle
import winoground
import vl_checklist
from src.random_augmenter import random_augment, RandomAugmentation

parser = argparse.ArgumentParser(description="FT CLIP MODEL WITH SYNTHETIC")

# parser.add_argument('--backbone_name', default='RN50', type=str, help='either of RN50, RN50x64, ViT-B/32')
parser.add_argument(
    "--load_from_path",
    default="",
    type=str,
    help="filename to load from directory path",
)
parser.add_argument("--steps", default=25, type=int, help="load from step/epoch")
parser.add_argument(
    "--is_lora", default=False, action="store_true", help="Is using lora"
)
parser.add_argument(
    "--heavy_aug",
    default=False,
    action="store_true",
    help="Is using heavy data augmentation",
)
parser.add_argument(
    "--benchmark",
    default="winoground",
    choices=["winoground", "vl_checklist"],
    help="benchmarks: winoground, vl_checklist",
)
parser.add_argument(
    "--save_as", default="", type=str, help="output name attached to pickle file"
)
parser.add_argument(
    "--filter_by",
    default="_",
    type=str,
    help="additional description to filter ckpt file - e.g., date: 2022-09-14",
)

if __name__ == "__main__":
    args = parser.parse_args()
    all_tests_config = {}
    if args.benchmark == "winoground":
        benchmark = winoground.WINOGROUND()
    elif args.benchmark == "vl_checklist":
        benchmark = vl_checklist.VLChecklist(
            gentype_="Relation", fromset_="vg", dtype_="spatial"
        )
        benchmark_2 = vl_checklist.VLChecklist(
            gentype_="Attribute", fromset_="vg", dtype_="color"
        )

    for current_arch in ["RN50", "RN50x64", "ViT-B32"]:
        all_files = glob.glob(args.load_from_path + f"/{current_arch}" + "/*.ckpt")
        search_epochs = "_" + str(args.steps) + "epochs"

        if current_arch == "ViT-B32":
            current_arch = "ViT-B/32"
        all_tests_config[current_arch] = []

        for f_name in all_files:
            if search_epochs in f_name and args.filter_by in f_name:
                print(current_arch, f_name)

                model, preprocess = benchmark.set_model(
                    model_type=current_arch, path=f_name, lora=args.is_lora
                )
                if args.heavy_aug:
                    preprocess.transforms.insert(
                        0, RandomAugmentation(random_augment())
                    )  # adding random_augmentations
                if args.benchmark == "winoground":
                    set_res1, set_res2 = benchmark.eval(model, preprocess)
                    all_tests_config[current_arch].append(
                        {
                            "f_name": f_name,
                            "full_set_res": set_res1,
                            "position_set_res": set_res2,
                        }
                    )
                elif args.benchmark == "vl_checklist":
                    gentype_1 = "Relation"
                    fromset_1 = "vg"
                    dtype_1 = "spatial"
                    set_res1, _ = benchmark.eval(model, preprocess)
                    ## get color benchmark
                    gentype_2 = "Attribute"
                    fromset_2 = "vg"
                    dtype_2 = "color"
                    set_res2, _ = benchmark_2.eval(model, preprocess)
                    all_tests_config[current_arch].append(
                        {
                            "f_name": f_name,
                            f"{gentype_1}_{fromset_1}_{dtype_1}": set_res1,
                            f"{gentype_2}_{fromset_2}_{dtype_2}": set_res2,
                        }
                    )

                print("=======")

        if args.is_lora:
            pickle.dump(
                all_tests_config,
                open(
                    f"{args.benchmark}_results_with_lora_{args.steps}steps_{args.save_as}.p",
                    "wb",
                ),
            )
        else:
            pickle.dump(
                all_tests_config,
                open(
                    f"{args.benchmark}_results_{args.steps}steps_{args.save_as}.p", "wb"
                ),
            )
