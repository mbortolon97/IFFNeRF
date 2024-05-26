import argparse
import functools
import json
import os
from functools import partial
from typing import Callable, Union, Tuple, List, Dict, Optional

import numpy as np
import torch

from dataLoader.blender import BlenderDataset
from dataLoader.tankstemple import TanksTempleDataset
from inerf.estimate_pose_inerf import pose_estimation as pose_estimation_inerf

import traceback
from pose_estimation.args import parse_args
from pose_estimation.eval_utils import parse_exp_dir
from pose_estimation.identification_module import IdentificationModule
from pose_estimation.model_utils import load_model, explore_model
from pose_estimation.train import train_id_module
from pose_estimation.test import test_pose_estimation


def pretrain_single_object(
    args: argparse.Namespace,
    data_path: str,
    dataset: Union[type[BlenderDataset], type[TanksTempleDataset]],
    ckpt_path: str,
    sequence_id: str,
    category_name: str,
    device: str,
    pose_estimation_func,
    starting_seed: int,
    augmentation_parameters: Dict[str, float],
    inerf_refinement: bool = False,
    lock_backbone: bool = True,
):
    print("data_path: ", data_path)
    train_dataset = dataset(
        data_path, split="train", downsample=args.downsample_train, is_stack=True
    )
    test_dataset = dataset(
        data_path, split="test", downsample=args.downsample_train, is_stack=True
    )

    nerf_model = load_model(ckpt_path, device)

    backbone_type = "dino"
    id_module = (
        IdentificationModule(backbone_type=backbone_type)
        .to(device, non_blocking=True)
        .train()
    )

    if lock_backbone:
        for parameter in id_module.image_preprocessing_net.parameters():
            parameter.require_grad = False

    start_iterations = 0
    ckpt_dirpath = os.path.dirname(ckpt_path)
    id_module_ckpt_path = os.path.join(ckpt_dirpath, "id_module.th")
    if os.path.exists(id_module_ckpt_path):
        print("Checkpoint already exist, skip training phase")
        ckpt_dict = torch.load(id_module_ckpt_path, map_location=device)
        id_module.load_state_dict(ckpt_dict["model_state_dict"])
        start_iterations = ckpt_dict["epoch"]

    if augmentation_parameters["resampling"]:
        generator_callable = functools.partial(explore_model, nerf_model)
    else:
        rays_ori, rays_dirs, rays_rgb = explore_model(nerf_model)
        generator_callable = lambda: (rays_ori, rays_dirs, rays_rgb)

    train_id_module(
        id_module_ckpt_path,
        device,
        id_module,
        generator_callable,
        train_dataset,
        test_dataset,
        sequence_id,
        start_iterations=start_iterations,
        # inerf_refinement=inerf_refinement,
        nerf_model=nerf_model,
        lock_backbone=lock_backbone,
    )

    print("Training complete starting testing phase...")
    print("Testing overfit performances...")
    rays_ori, rays_dirs, rays_rgb = explore_model(nerf_model)

    model_up = torch.mean(train_dataset.poses[:, :3, 1], dim=0).to(
        device=rays_ori.device
    )

    print("Testing performances on same points...")

    np.random.seed(starting_seed)
    torch.manual_seed(starting_seed)
    
    (
        _,
        val_avg_translation_error,
        val_avg_angular_error,
        val_avg_score,
        val_recall,
    ) = test_pose_estimation(
        test_dataset,
        id_module,
        rays_ori,
        rays_dirs,
        rays_rgb,
        model_up,
        sequence_id=sequence_id,
        inerf_refinement=inerf_refinement,
        nerf_model=nerf_model,
        save=False,
        save_all=False,
        augmentation_parameters=augmentation_parameters,
    )

    print("Val AVG translation error: ", val_avg_translation_error)
    print("Val AVG angular error: ", val_avg_angular_error)
    print("Val AVG score error: ", val_avg_score)
    print("Val recall: ", val_recall)

    np.random.seed(starting_seed)
    torch.manual_seed(starting_seed)

    print("Testing real performances on real data...")
    rays_ori, rays_dirs, rays_rgb = explore_model(nerf_model)
    (
        test_results,
        test_avg_translation_error,
        test_avg_angular_error,
        test_avg_score,
        test_recall,
    ) = test_pose_estimation(
        test_dataset,
        id_module,
        rays_ori,
        rays_dirs,
        rays_rgb,
        model_up,
        sequence_id=sequence_id,
        # inerf_refinement=inerf_refinement,
        nerf_model=nerf_model,
        augmentation_parameters=augmentation_parameters,
    )

    print("Test AVG translation error: ", test_avg_translation_error)
    print("Test AVG angular error: ", test_avg_angular_error)
    print("Test AVG score error: ", test_avg_score)
    print("Test recall: ", test_recall)

    return test_results




def evaluate_single_object_in_blender(
    out_path: str,
    data_path: str,
    dataset_type: Union[type[BlenderDataset], type[TanksTempleDataset]],
    object_id: str,
    ckpt_path: str,
    args: argparse.Namespace,
    pose_estimation_func: Callable[
        [torch.Tensor, Union[np.ndarray, torch.Tensor], torch.Tensor, torch.nn.Module],
        Tuple[float, torch.Tensor, List[torch.Tensor]],
    ],
    augmentation_parameters: Dict[str, Union[float, str]],
    starting_seed: int = 55176280,
    device: str = "cuda",
    inerf_refinement: bool = False,
    lock_backbone: bool = True,
):
    # Explore given directory
    results = pretrain_single_object(
        args,
        data_path,
        dataset_type,
        ckpt_path,
        object_id,
        os.path.splitext(os.path.basename(out_path))[0],
        device,
        pose_estimation_func,
        starting_seed,
        augmentation_parameters,
        inerf_refinement=inerf_refinement,
        lock_backbone=lock_backbone,
    )

    if partial:
        print("Not all the sequences available")

    return results


def main():
    args, extras = parse_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    n_gpus = len(args.gpu.split(","))

    if args.algorithm_type == "inerf":
        pose_estimation_func = partial(pose_estimation_inerf, print_progress=False)
    elif args.algorithm_type == "inerf_dice":
        pose_estimation_func = partial(
            pose_estimation_inerf, dice_loss=True, print_progress=False
        )
    else:
        raise ValueError("unknown algorithm")

    augmentation_parameters = {
        "resampling": True,
    }

    # create destination directory structure
    out_path_abs = os.path.abspath(args.out_path)
    out_dir = os.path.dirname(out_path_abs)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.data_type == "blender":
        dataset_type = BlenderDataset
        suffix = "_VM"
    else:
        dataset_type = TanksTempleDataset
        suffix = "_VMtt"

    # create the container for the results
    results = []
    experiments_to_test = parse_exp_dir(args.exp_patch, suffix)
    for experiment_to_test in experiments_to_test.values():
        checkpoint_filepath = experiment_to_test["checkpoint_filepath"]
        object_id = experiment_to_test["sequence_id"]
        data_path = os.path.join(args.datadir, object_id)
        try:
            obj_results = evaluate_single_object_in_blender(
                args.out_path,
                data_path,
                dataset_type,
                object_id,
                checkpoint_filepath,
                args,
                pose_estimation_func,
                augmentation_parameters,
                starting_seed=55176280,
                device=device,
                inerf_refinement=False,
                lock_backbone=False,
            )

            results.extend(obj_results)
        except RuntimeError:
            traceback.print_exc()

    print("Saving results")
    with open(out_path_abs, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    torch.manual_seed(500661008)
    main()
