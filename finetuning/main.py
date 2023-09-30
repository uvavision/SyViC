import argparse
import src.core as core
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="FT CLIP MODEL WITH SYNTHETIC")

    parser.add_argument(
        "--backbone_name",
        default="RN50",
        type=str,
        help="either of RN50, RN50x64, ViT-B/32",
    )
    parser.add_argument(
        "--folder_name",
        default="",
        type=str,
    )
    parser.add_argument(
        "--load_from_path", default="", type=str, help="filename to load from"
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=0,
        type=int,
        help="define seed for random distribution of dataset",
    )
    parser.add_argument(
        "--frozen", dest="frozen", action="store_true", help="if freeze visual head"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=5e-7,
        type=float,
        metavar="LR",
        help="max learning rate",
    )  # for LoRA use much higher, like 0.0025
    parser.add_argument(
        "-b",
        "--batch_size",
        default=200,
        type=int,
        metavar="N",
        help="mini-batch size (default: 100)",
    )
    parser.add_argument(
        "--workers", default=8, type=int, help="number of cpus for data loading"
    )

    parser.add_argument(
        "--steps", default=201, type=int, help="number of cpus for data loading"
    )
    parser.add_argument(
        "--early_stop", default=201, type=int, help="number of cpus for data loading"
    )

    parser.add_argument(
        "--lora_r", default=-1, type=int, help="use any number above 0 to activate LoRA"
    )
    parser.add_argument(
        "--style_transfer",
        default=None,
        type=str,
        help="Style transfer mechanism. Currently supported are: 'adain'",
    )
    parser.add_argument(
        "--mixstyle",
        default=False,
        action="store_true",
        help="Use mixstyle for training real life + synthetic data",
    )
    parser.add_argument(
        "--capsplit",
        default=False,
        action="store_true",
        help="Use caption splitting for training",
    )
    parser.add_argument(
        "--save_every", default=1, type=int, help="number of cpus for data loading"
    )

    parser.add_argument(
        "--local_rank",
        default=os.environ.get("LOCAL_RANK", 0),
        type=int,
        help="Please ignore and do not set this argument.",
    )
    parser.add_argument("--debug", action="store_true", help="do debug")
    parser.add_argument("--debug_port", default=12345, type=int, help="for debug")
    parser.add_argument("--num_workers", default=8, type=int, help="for number of cpus")
    parser.add_argument("--debug_addr", type=str, help="for debug")

    parser.add_argument(
        "--data_type",
        default="position",
        type=str,
        help="types of generated data: position, color, size, material, action, or any combination thereof",
    )
    parser.add_argument(
        "--heavy_aug",
        default=False,
        action="store_true",
        help="Use heavy data augmentation",
    )
    parser.add_argument(
        "--sample_fps",
        default=6.0,
        type=float,
        help="Number of sampled frames per second for data loading",
    )
    parser.add_argument(
        "--dataset_subset",
        default=1.0,
        type=float,
        help="Portion of the subset of the dataset to be used for training (e.g. 0.1 means train on %10 of the data)",
    )
    parser.add_argument(
        "--action_captions",
        default="grammar_full",
        type=str,
        help="Which caption generation mechanism to use for the action dataset {'grammar', 'grammar_full', 'bloomz', 'flan-t5'}",
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        metavar="B1",
        help="betas first hyperparameter",
    )
    parser.add_argument(
        "--beta2",
        default=0.98,
        type=float,
        metavar="B2",
        help="betas second hyperparameter",
    )
    parser.add_argument(
        "--num_videos",
        default=-1,
        type=int,
        help="Number of videos to use in training from `videos_objects_ascending.json` for diversity sampling. Specify `-1` to disable diversity sampling.",
    )
    parser.add_argument(
        "--eps", default=1e-6, type=float, metavar="EPS", help="epsilon hyperparameter"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.001,  # For lora, use much higher, like 0.2
        type=float,
        metavar="WD",
        help="weight decay hyperparameter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Finetune
    train_job = core.FT_CLIP(args)
