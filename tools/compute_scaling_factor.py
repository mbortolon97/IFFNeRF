import configargparse
import argparse
import os
from typing import Union

import torch

from dataLoader.blender import BlenderDataset
from dataLoader.mip360 import Mip360Dataset
from dataLoader.tankstemple import TanksTempleDataset
from opt import build_argparse
from models.tensoRF import TensorCP, TensorVMSplit


def parse_args():
    original_parser = build_argparse()
    parser = configargparse.ArgumentParser(parents=[original_parser], add_help=False)
    parser.add_argument(
        "--exp_patch",
        type=str,
        required=True,
        default="./log",
        help="experiment directory",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["blender", "tankstemple", "mip360"],
        default="mip360",
        help="the type of data to validate",
    )

    args, extras = parser.parse_known_args()
    return args, extras


def get_highest_valid_checkpoint(root_dir):
    sorted_filenames = sorted(os.listdir(root_dir), reverse=True)
    for file_name in sorted_filenames:
        ckpt_filepath = os.path.join(root_dir, file_name)
        if os.path.isfile(ckpt_filepath) and ckpt_filepath.endswith(".th"):
            return ckpt_filepath

    return ""


def parse_exp_dir(exp_dir, suffix):
    objects_checkpoints = {}
    exp_dirs_filenames = os.listdir(exp_dir)
    for exp_dir_filename in exp_dirs_filenames:
        exp_dir_filepath = os.path.join(exp_dir, exp_dir_filename)
        if not (
            os.path.isdir(exp_dir_filepath)
            and exp_dir_filename.startswith("tensorf_")
            and exp_dir_filename.endswith(suffix)
        ):
            continue
        name_components = exp_dir_filepath.split("_")
        sequence_id = name_components[-2]
        category_name = ""
        checkpoint_filepath = get_highest_valid_checkpoint(exp_dir_filepath)
        if checkpoint_filepath == "":
            print(
                f"Object {sequence_id} of category {category_name} skipped because no valid checkpoint found"
            )
            continue
        objects_checkpoints[sequence_id] = {
            "checkpoint_filepath": checkpoint_filepath,
            "sequence_id": sequence_id,
            "category_name": category_name,
        }
    return objects_checkpoints


def load_model(checkpoint_path, device):
    # Load TensoRF
    ckpt = torch.load(checkpoint_path, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(ckpt["model_name"])(**kwargs)
    tensorf.load(ckpt)

    for param in tensorf.parameters():
        param.requires_grad = False
    return tensorf


def compute_single_object_scaling_factor(
    args: argparse.Namespace,
    data_path: str,
    dataset: Union[type[BlenderDataset], type[TanksTempleDataset]],
    ckpt_path: str,
    sequence_id: str,
):
    results = []

    print("data_path: ", data_path)

    # points = torch.tensor([0., 0., 0., 1.], dtype=test_dataset.poses.dtype, device=test_dataset.poses.device)[None, :, None].expand(test_dataset.poses.shape[0], -1, -1)
    # camera_positions = torch.bmm(test_dataset.poses, points)
    model = load_model(ckpt_path, 'cpu')

    bb_size = model.aabb[1] - model.aabb[0]
    max_bb_size = torch.max(bb_size)
    aabb_to_unit_box_ratio = 1. / max_bb_size
    print(f"[{sequence_id}] aabb_to_unit_box_ratio: {aabb_to_unit_box_ratio}")

    return results


def main():
    args, extras = parse_args()
    
    if args.data_type == "blender":
        dataset_type = BlenderDataset
        suffix = "_VM"
    elif args.data_type == "mip360":
        dataset_type = Mip360Dataset
        suffix = "_VMmip"
    else:
        dataset_type = TanksTempleDataset
        suffix = "_VMtt"

    # create the container for the results
    experiments_to_test = parse_exp_dir(args.exp_patch, suffix)
    for experiment_to_test in experiments_to_test.values():
        checkpoint_filepath = experiment_to_test["checkpoint_filepath"]
        object_id = experiment_to_test["sequence_id"]
        data_path = os.path.join(args.datadir, object_id)
        compute_single_object_scaling_factor(
            args,
            data_path,
            dataset_type,
            checkpoint_filepath,
            object_id,
        )


if __name__ == "__main__":
    main()
