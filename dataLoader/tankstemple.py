import math

import torch
import torchshow
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T

from dataLoader.spiral_utils import create_spiral, make_look_at
from ray_utils import get_ray_directions_Ks, get_rays
import numpy as np


def circle(radius=3.5, h=0.0, axis="z", t0=0, r=1):
    if axis == "z":
        return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
    elif axis == "y":
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]


def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return (x / l2,)


def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)


def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R


def gen_path(pos_gen, at=(0, 0, 0), up=(0, -1, 0), frames=180):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(t * (360.0 / frames) / 180 * np.pi))
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)


class TanksTempleDataset(Dataset):
    """NSVF Generic Dataset."""

    def __init__(
        self, datadir, split="train", downsample=1.0, ori_wh=None, is_stack=False
    ):
        self.transform = None
        if ori_wh is None:
            ori_wh = [1920, 1080]
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.img_wh = (int(ori_wh[0] / downsample), int(ori_wh[1] / downsample))
        self.downsample_ratio = (
            float(self.img_wh[0]) / float(ori_wh[0]),
            float(self.img_wh[1]) / float(ori_wh[1]),
        )
        self.define_transforms()

        self.white_bg = True
        self.near_far = [0.01, 6.0]
        self.scene_bbox = (
            torch.from_numpy(np.loadtxt(f"{self.root_dir}/bbox.txt"))
            .float()[:6]
            .view(2, 3)
            * 1.2
        )

        self.read_meta()
        self.define_proj_mat()

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):
        np_intrinsic = np.loadtxt(os.path.join(self.root_dir, "intrinsics.txt"))
        self.intrinsics = (
            torch.from_numpy(np_intrinsic)[:3, :3].to(dtype=torch.float32)
        )[None]
        self.intrinsics[
            :, :2
        ] /= (
            self.downsample
        )  # modify focal length and principal point to match size self.img_wh
        self.intrinsics = self.intrinsics.contiguous()

        pose_files = sorted(os.listdir(os.path.join(self.root_dir, "pose")))
        img_files = sorted(os.listdir(os.path.join(self.root_dir, "rgb")))

        if self.split == "train":
            pose_files = [x for x in pose_files if x.startswith("0_")]
            img_files = [x for x in img_files if x.startswith("0_")]
        elif self.split == "val":
            pose_files = [x for x in pose_files if x.startswith("1_")]
            img_files = [x for x in img_files if x.startswith("1_")]
        elif self.split == "test":
            test_pose_files = [x for x in pose_files if x.startswith("2_")]
            test_img_files = [x for x in img_files if x.startswith("2_")]
            if len(test_pose_files) == 0:
                test_pose_files = [x for x in pose_files if x.startswith("1_")]
                test_img_files = [x for x in img_files if x.startswith("1_")]
            pose_files = test_pose_files
            img_files = test_img_files

        # ray directions for all pixels, same for all images (same H, W, focal)
        w, h = self.img_wh
        self.ori_directions, dx, dy = get_ray_directions_Ks(h, w, self.intrinsics)
        self.directions = self.ori_directions / torch.norm(
            self.ori_directions, dim=-1, keepdim=True
        )
        dx = dx.contiguous()
        dy = dy.contiguous()

        self.K = self.intrinsics

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []

        assert len(img_files) == len(pose_files)
        for img_fname, pose_fname in tqdm(
            zip(img_files, pose_files),
            desc=f"Loading data {self.split} ({len(img_files)})",
        ):
            c2w = np.loadtxt(
                os.path.join(self.root_dir, "pose", pose_fname)
            )  # @ cam_trans
            c2w = torch.tensor(c2w, dtype=torch.float32).contiguous()
            self.poses.append(c2w)  # C2W

            image_path = os.path.join(self.root_dir, "rgb", img_fname)
            self.image_paths.append(image_path)
            img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.permute(1, 2, 0)  # (h, w, 4) RGBA
            if img.shape[-1] == 3:
                default_bkg = torch.ones((1, 1, 3), dtype=img.dtype, device=img.device)
                distance = torch.linalg.norm(img - default_bkg, dim=-1)
                mask = torch.logical_not(distance < (5.0 / 255.0)).to(dtype=img.dtype)
                img = torch.cat((img, mask[..., None]), dim=-1)
            self.all_rgbs.append(img)

            rays_o, rays_d, radii = get_rays(
                self.directions,
                c2w,
                directions=self.ori_directions,
                dx=dx,
                dy=dy,
                keepdim=True,
            )  # both (h*w, 3)

            self.all_rays.append(torch.cat([rays_o, rays_d, radii], -1))  # (h*w, 7)

        self.poses = torch.stack(self.poses)

        train_cam_points = (
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=self.poses.dtype, device=self.poses.device
            ).reshape(1, 1, 4)
            @ self.poses[..., :3, :].mT
        )[..., 0, :]

        # subtract out the centroid and take the SVD
        camera_center_points = torch.mean(train_cam_points, dim=-1, keepdim=True)
        svd = torch.linalg.svd(camera_center_points - train_cam_points)

        # Extract the left singular vectors
        left = svd[0]
        plane_normal = left[:, -1]

        center_point = (self.scene_bbox[0] + self.scene_bbox[1]) / 2.0
        center_distances = torch.linalg.norm(train_cam_points - center_point, dim=-1)
        avg_center_distance = torch.mean(center_distances)

        up, _ = normalize(self.poses[:, :3, 1].sum(0))

        theta_max = 2 * math.pi
        theta = torch.linspace(
            0, theta_max, 100, dtype=self.poses.dtype, device=self.poses.device
        )
        r = avg_center_distance * 1.4
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z_train_cameras = torch.mean(train_cam_points, dim=0)[-1]
        z = torch.full(
            (100,),
            z_train_cameras.item(),
            dtype=self.poses.dtype,
            device=self.poses.device,
        )

        camera_positions = torch.stack((y, z, x), dim=-1)

        camera_positions = camera_positions + center_point

        c2ws = []
        for camera_position in camera_positions:
            c2w = make_look_at(camera_position, center_point, up)
            c2ws.append(c2w)
        self.render_path = torch.stack(c2ws)

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
            )  # (len(self.meta['frames]),h, w, 3)
            self.all_rgbs = torch.stack(
                self.all_rgbs, 0
            )  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = (
            self.intrinsics[:3, :3].unsqueeze(0) @ torch.inverse(self.poses)[:, :3]
        )

    def world2ndc(self, points):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

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
