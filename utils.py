import math

import cv2
import numpy as np
import scipy.signal
import torch
import torchvision.transforms as T
from PIL import Image

mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
import random


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    import lpips

    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[: len(target)] == target:
            return one
    return None


""" Evaluation metrics (ssim, lpips)
"""


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def power_transformation(centered_xyz: torch.Tensor, alpha: float = -1.5):
    x_abs = torch.abs(centered_xyz)
    negate_alpha = math.fabs(alpha - 1)
    return (
        torch.sign(centered_xyz)
        * (negate_alpha / alpha)
        * (torch.pow((x_abs / negate_alpha) + 1.0, alpha) - 1.0)
    )


def power_transformation_inv(center_metric: torch.Tensor, alpha: float = -1.5):
    negate_alpha = math.fabs(alpha - 1)
    return (
        torch.sign(center_metric)
        * (
            torch.pow(
                ((alpha * torch.abs(center_metric)) + negate_alpha) / negate_alpha,
                1 / alpha,
            )
            - 1.0
        )
        * negate_alpha
    )


import plyfile
import skimage.measure


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1] - bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[..., ::-1]  # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0, 0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0, 1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0, 2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flops_to_string(flops):
    suffixes = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs", "EFLOPs"]

    if flops < 0:
        raise ValueError("FLOPs value must be non-negative.")

    if flops == 0:
        return "0 FLOPs"

    i = 0
    while flops >= 1000 and i < len(suffixes) - 1:
        flops /= 1000
        i += 1

    return "{:.2f} {}".format(flops, suffixes[i])


def format_time(milliseconds):
    if milliseconds < 0:
        raise ValueError("Time value must be non-negative.")

    if milliseconds < 1000:
        return f"{milliseconds} ms"

    seconds = milliseconds / 1000
    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} min"

    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f} hours"

    days = hours / 24
    return f"{days:.2f} days"
