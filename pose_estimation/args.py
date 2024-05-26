import configargparse
from opt import build_argparse

def parse_args():
    original_parser = build_argparse()
    parser = configargparse.ArgumentParser(parents=[original_parser], add_help=False)
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")
    parser.add_argument(
        "--resume", default=None, help="path to the weights to be resumed"
    )
    parser.add_argument(
        "--exp_patch",
        type=str,
        required=True,
        default="./log",
        help="experiment directory",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        default="pose_eval.json",
        help="experiment directory",
    )
    parser.add_argument(
        "--resize_factor",
        type=float,
        default=1.0,
    )
    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    # algorithm type
    parser.add_argument(
        "--algorithm_type",
        type=str,
        default="inerf",
    )
    # algorithm type
    parser.add_argument(
        "--starting_pose_strategy",
        type=str,
        default="histogram_comparison",
    )
    # evaluation_shortcuts
    parser.add_argument(
        "--limit_categories",
        type=str,
        nargs="+",
        default=[],
    )

    args, extras = parser.parse_known_args()
    return args, extras