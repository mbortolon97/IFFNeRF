import numpy as np
from statistics import mean, stdev
import pandas as pd
import os
import torch
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef

# inject the path to the root of the project
import sys

code_directory = os.path.dirname(os.path.dirname(__file__))
print(code_directory)
sys.path.append(code_directory)

from dataLoader.blender import BlenderDataset
from evaluate_lego import build_hist
import matplotlib.pyplot as plt


def compute_correlation(datadir: str, comparison="pearson"):
    train_dataset = BlenderDataset(datadir, split='train', downsample=1., is_stack=True)
    train_hists = build_hist(train_dataset.all_rgbs.to(device='cuda'))

    # bins = torch.torch.linspace(0, train_dataset.all_rgbs.max(), 255, device=train_dataset.all_rgbs.device)
    # plt.bar(bins.cpu().numpy(), train_hists[50].cpu().numpy(), width=(train_dataset.all_rgbs.max() / 255.).item())
    # plt.show()

    test_dataset = BlenderDataset(datadir, split='test', downsample=1., is_stack=True)
    test_hists = build_hist(test_dataset.all_rgbs.to(device='cuda'))

    train_poses = train_dataset.poses.to(device='cuda')
    train_camera_position = torch.bmm(
        torch.tensor(
            [0., 0., 0., 1.], dtype=train_poses.dtype, device=train_poses.device
        )[None].expand(train_poses.shape[0], -1, -1),
        torch.transpose(train_poses[:, :3, :], -1, -2)
    )[:, 0]

    test_poses = test_dataset.poses.to(device='cuda')
    test_camera_position = torch.bmm(
        torch.tensor(
            [0., 0., 0., 1.], dtype=test_poses.dtype, device=test_poses.device
        )[None].expand(test_poses.shape[0], -1, -1),
        torch.transpose(test_poses[:, :3, :], -1, -2)
    )[:, 0]
    relative_distances = torch.cdist(train_camera_position, test_camera_position)

    train_coords, test_coords = torch.meshgrid(
        torch.arange(0, train_dataset.poses.shape[0], dtype=torch.int64, device=train_dataset.poses.device),
        torch.arange(0, test_dataset.poses.shape[0], dtype=torch.int64, device=test_dataset.poses.device),
        indexing='ij'
    )

    if comparison == 'spearman':
        coefficients = spearman_corrcoef(
            train_hists[train_coords.reshape(-1)].permute(1, 0), test_hists[test_coords.reshape(-1)].permute(1, 0)
        ).reshape(*train_coords.shape)
    elif comparison == 'pearson':
        coefficients = pearson_corrcoef(
            train_hists[train_coords.reshape(-1)].permute(1, 0), test_hists[test_coords.reshape(-1)].permute(1, 0)
        ).reshape(*train_coords.shape)
    elif comparison == 'chi_squared':
        ori_hist = train_hists[train_coords.reshape(-1)]
        test_hist = test_hists[test_coords.reshape(-1)]
        coefficients = 0.5 * torch.sum(torch.square(ori_hist - test_hist) / (ori_hist + test_hist), dim=-1)
        coefficients = coefficients.reshape(*train_coords.shape)
    elif comparison == 'L2':
        ori_hist = train_hists[train_coords.reshape(-1)]
        test_hist = test_hists[test_coords.reshape(-1)]
        coefficients = torch.sqrt(torch.sum(torch.square(ori_hist - test_hist), dim=-1))
        coefficients = coefficients.reshape(*train_coords.shape)
    elif comparison == 'kl_divergence':
        coefficients = torch.nn.functional.kl_div(
            train_hists[train_coords.reshape(-1)], test_hists[test_coords.reshape(-1)], reduction='none').sum(dim=-1).reshape(
            *train_coords.shape)
        print(coefficients.shape)
    else:
        raise ValueError("Unknown comparison method")

    # if comparison in ['spearman', 'pearson']:
    #     nearest_idx = coefficients.max(dim=0).indices
    # else:
    #     nearest_idx = coefficients.min(dim=0).indices
    # selected_coefficient = torch.gather(coefficients, 0, nearest_idx.reshape(1, -1))
    # selected_distances = torch.gather(relative_distances, 0, nearest_idx.reshape(1, -1))

    # plt.scatter(selected_distances.view(-1).cpu().numpy(), selected_coefficient.view(-1).cpu().numpy())
    plt.scatter(relative_distances.view(-1).cpu().numpy(), coefficients.view(-1).cpu().numpy())
    plt.suptitle(f"Correlation results using the {comparison} metric", fontsize=20)
    plt.xlabel('Geodetic distance')
    plt.ylabel('Correlation')

    plt.xlim([0., 5.])
    if comparison in ['spearman', 'pearson']:
        plt.ylim([.5, 1.])
    plt.show()


if __name__ == "__main__":
    compute_correlation("/home/mbortolon/data/datasets/nerf_synthetic/lego", comparison='chi_squared')
