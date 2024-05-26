import os.path
from functools import lru_cache

import torch
from tqdm import tqdm

from ray_utils import get_ray_directions_Ks, get_rays
from .utils import downsample, recenter_poses, rescale_poses

from .spiral_utils import create_spiral
from .llff import normalize
from dataLoader.colmap_utils import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
    qvec2rotmat,
)
from PIL import Image
import numpy as np
import os
import sys
from torchvision import transforms as T


def readColmapCameras(
    cam_extrinsics, cam_intrinsics, images_folder, resize_factor: float = 1.0
):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = focal_length_x
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE "
                "cameras) supported!"
            )

        intrinsic_matrix = np.asarray(
            [
                [focal_length_x / resize_factor, 0.0, cx / resize_factor],
                [0.0, focal_length_y / resize_factor, cy / resize_factor],
                [0.0, 0.0, 1.0],
            ]
        )

        w2c = np.eye(4, dtype=R.dtype)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))

        camera = {
            "uid": uid,
            "K": intrinsic_matrix,
            "cam2world": c2w,
            "image_filename": os.path.abspath(image_path),
        }

        cam_infos.append(camera)

    return cam_infos


@lru_cache
def read_annotations(data_dir: str, resize_factor: float = 1.0):
    try:
        cameras_extrinsic_file = os.path.join(data_dir, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(data_dir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(data_dir, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(data_dir, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(data_dir, "images"),
        resize_factor=resize_factor,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x["uid"])

    bin_path = os.path.join(data_dir, "sparse/0/points3D.bin")
    txt_path = os.path.join(data_dir, "sparse/0/points3D.txt")

    try:
        xyz, _, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)

    cam2worlds = []
    intrinsics = []
    images_filenames = []
    for cam_info in cam_infos:
        intrinsics.append(cam_info["K"])
        cam2worlds.append(cam_info["cam2world"])
        images_filenames.append(cam_info["image_filename"])

    cam2worlds = np.stack(cam2worlds, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    cam2worlds, inv_transformation = recenter_poses(
        cam2worlds
    )  # original , method='pca'
    pcd_transformation = np.linalg.inv(inv_transformation)
    pcd_transform = np.concatenate(
        (xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)), axis=-1
    )
    centered_pcd = pcd_transformation @ pcd_transform.T
    size = np.abs(centered_pcd).max(axis=0)
    max_size = size.max()

    cam2worlds, inv_scale = rescale_poses(cam2worlds, max_size)

    return cam2worlds, intrinsics, images_filenames, inv_scale, inv_transformation


def _load_renderings(data_dir: str, split: str, resize_factor: float = 1.0, hold_out=8):
    (
        cam2worlds,
        intrinsics,
        images_filenames,
        inv_scale,
        inv_transformation,
    ) = read_annotations(data_dir, resize_factor=resize_factor)

    # split selection
    all_indices = np.arange(cam2worlds.shape[0])
    if split == "train":
        split_index = all_indices[all_indices % hold_out != 0]
    elif split == "test" or split == "val":
        split_index = all_indices[all_indices % hold_out == 0]
    else:
        raise ValueError("invalid split type")
    # All per-image quantities must be re-indexed using the split indices.
    split_cam2worlds = cam2worlds[split_index]
    split_intrinsics = intrinsics[split_index]
    split_images_filenames = [images_filenames[split_idx] for split_idx in split_index]

    return (
        split_images_filenames,
        split_cam2worlds,
        split_intrinsics,
        inv_transformation,
        inv_scale,
    )


class Mip360Dataset(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]

    def __init__(
        self,
        datadir: str,
        split: str = "train",
        downsample: float = 4.0,
        is_stack=False,
        color_bkgd_aug: str = "white",
        n_test_interpolation: int = 0,
        N_vis=-1,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        self.split = split
        self.N_vis = N_vis

        (
            images_filenames,
            camtoworlds_np,
            _K,
            inv_transformation,
            inv_scale,
        ) = _load_renderings(datadir, split, resize_factor=downsample)
        self.training = split in ["train", "trainval"]
        self.K = torch.from_numpy(_K).to(torch.float32)  # (3, 3)
        self.intrinsics = self.K
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        img = Image.open(images_filenames[0])
        self.width = img.width
        self.height = img.height
        self.img_wh = (self.width, self.height)
        self.downsample = downsample
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.white_bg = False
        self.near_far = [0.01, 1.4]
        self.img_wh = (
            int(self.width / downsample),
            int(self.height / downsample),
        )
        self.focal = [self.K[0, 0, 0], self.K[0, 1, 1]]

        self.ori_directions, dx, dy = get_ray_directions_Ks(
            self.img_wh[1], self.img_wh[0], self.intrinsics[[0]]
        )
        self.directions = self.ori_directions / torch.norm(
            self.ori_directions, dim=-1, keepdim=True
        )

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []

        img_eval_interval = 1 if self.N_vis < 0 else len(images_filenames) // self.N_vis
        idxs = list(range(0, len(images_filenames), img_eval_interval))
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#
            pose = camtoworlds_np[i]
            c2w = torch.tensor(pose, dtype=torch.float32)
            self.poses.append(c2w)

            self.image_paths.append(images_filenames[i])
            img = Image.open(images_filenames[i])

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.permute(1, 2, 0)  # (h*w, 4) RGBA
            # img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d, radii = get_rays(
                self.directions,
                c2w,
                directions=self.ori_directions,
                dx=dx,
                dy=dy,
                keepdim=True,
            )  # both (h*w, 3)

            self.all_rays.append(
                torch.cat([rays_o[0], rays_d[0], radii[0]], -1)
            )  # (h*w, 7)

        # self.is_stack = True

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0).reshape(
                -1, self.all_rays[-1].shape[-1]
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0).reshape(
                -1, self.all_rgbs[-1].shape[-1]
            )  # (len(self.meta['frames])*h*w, 3)
            # self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(
                self.all_rays, 0
            )  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(
                self.all_rgbs, 0
            )  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

        # from visual import display_camera_rays
        # display_camera_rays(
        #     self.poses[..., :3, :],
        #     rays_ori=self.all_rays[..., :3],
        #     rays_dir=self.all_rays[..., 3:6],
        #     scale=0.01,
        # )

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}

        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]

            sample = {"rays": rays, "rgbs": img}
        return sample
