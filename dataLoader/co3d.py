"""
Copyright (c) 2022 Matteo Bortolon
"""

import os
import json
from functools import lru_cache

from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset

import pytorch_lightning as pl

from .rays import get_rays_for_each_pixel, get_rays
from .llff import normalize

import os

import imageio.v3 as iio
import numpy as np
import torch

from .utils import downsample, recenter_poses, rescale_poses

from cv2 import INTER_LANCZOS4, INTER_LINEAR
from .spiral_utils import create_spiral
from typing import List, Dict
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation
)

VAL_SPLIT_EVERY = 10


def load_images_annotation(data_dir: str, split: str):
    assert split in ["train", "val", "test"], "split is not valid"
    assert os.path.isdir(data_dir), "data_dir should be a directory"

    abs_data_dir = os.path.abspath(data_dir)
    sequence_name = os.path.basename(abs_data_dir.rstrip('/'))
    category_dir = os.path.abspath(os.path.join(abs_data_dir, ".."))
    co3d_root = os.path.abspath(os.path.join(category_dir, ".."))

    imgs_arrays, inv_scale, inv_transformation = read_category_annotations(category_dir, sequence_name)

    # load in memory data
    sequence_imgs: List[FrameAnnotation] = imgs_arrays[split]
    assert len(sequence_imgs) != 0, f"Selected split {split} is empty"
    for frame_annotation in sequence_imgs:
        if not os.path.isabs(frame_annotation.image.path):
            frame_annotation.image.path = os.path.join(co3d_root, frame_annotation.image.path)
        if frame_annotation.depth is not None:
            if not os.path.isabs(frame_annotation.depth.path):
                frame_annotation.depth.path = os.path.join(co3d_root, frame_annotation.depth.path)
            if frame_annotation.depth.mask_path is not None:
                if not os.path.isabs(frame_annotation.depth.mask_path):
                    frame_annotation.depth.mask_path = os.path.join(co3d_root, frame_annotation.depth.mask_path)
        if frame_annotation.mask is not None:
            frame_annotation.mask.path = os.path.join(co3d_root, frame_annotation.mask.path)

    return sequence_imgs, inv_transformation, inv_scale


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def load_depth(path: str, scale_adjustment: float) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth file name "%s"' % path)

    d = load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[..., None]  # fake feature channel


def load_1bit_png_mask(file: str) -> np.ndarray:
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def load_depth_mask(path: str) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth mask file name "%s"' % path)
    m = load_1bit_png_mask(path)
    return m[..., None]  # fake feature channel


@lru_cache
def read_category_annotations(category_dir: str, sequence_name: str):
    category_frame_annotations: List[FrameAnnotation] = load_dataclass_jgzip(
        os.path.join(category_dir, "frame_annotations.jgz"), List[FrameAnnotation]
    )
    # Not used, leaved for reference
    # category_sequence_annotations: List[SequenceAnnotation] = load_dataclass_jgzip(
    #     os.path.join(category_dir, "sequence_annotations.jgz"), List[SequenceAnnotation]
    # )

    # Init container to store frame annotations
    imgs_arrays = {'train': [], 'test': [], 'val': []}

    # Read all the annotations
    train_set = set()
    val_set = set()
    test_set = set()
    set_list = os.path.join(category_dir, "set_lists")
    for set_file in os.listdir(set_list):
        set_filepath = os.path.join(set_list, set_file)
        if os.path.isfile(set_filepath):
            with open(os.path.join(set_list, set_file), 'r') as fh:
                set_data = json.load(fh)
                for train_entry in set_data["train"]:
                    if sequence_name == train_entry[0]:
                        train_set.add(train_entry[1])
                for val_entry in set_data["val"]:
                    if sequence_name == val_entry[0]:
                        val_set.add(val_entry[1])
                for test_entry in set_data["test"]:
                    if sequence_name == test_entry[0]:
                        test_set.add(test_entry[1])

    # transpose coordinates
    co3d2opencv = np.asarray(
        [[-1., 0., 0., 0.],
         [0., -1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]], dtype=np.float32)

    cam2worlds_idx = []
    cam2worlds = []
    intrinsics = []
    for frame_idx, frame_annotation in enumerate(category_frame_annotations):
        if frame_annotation.sequence_name != sequence_name:  # performance shortcut avoid considering frame of difference sequences
            continue
        mtx: np.ndarray = np.eye(4)
        mtx[:3, :3] = np.asarray(frame_annotation.viewpoint.R, dtype=np.float32)
        mtx[:3, -1] = np.asarray(frame_annotation.viewpoint.T, dtype=np.float32)
        # flip the XY coordinates.
        mtx = mtx @ co3d2opencv
        # permute the axis
        mtx[:3, :3] = np.transpose(mtx[:3, :3], (1, 0))

        scale = min(frame_annotation.image.size) / 2.0
        c0_width = frame_annotation.image.size[1] / 2.0
        c0_height = frame_annotation.image.size[0] / 2.0

        cx = -frame_annotation.viewpoint.principal_point[0] * scale + c0_width
        cy = -frame_annotation.viewpoint.principal_point[1] * scale + c0_height

        fx = -frame_annotation.viewpoint.focal_length[0] * scale
        fy = -frame_annotation.viewpoint.focal_length[1] * scale

        frame_annotation.viewpoint.principal_point = (cx, cy)
        frame_annotation.viewpoint.focal_length = (fx, fy)

        K_matrix = np.asarray([
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.],
        ])

        # w2c to c2w as in NeRF
        mtx = np.linalg.inv(mtx)
        cam2worlds.append(mtx)
        cam2worlds_idx.append(frame_idx)
        intrinsics.append(K_matrix)

    cam2worlds = np.stack(cam2worlds, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    cam2worlds, inv_transformation = recenter_poses(cam2worlds)
    cam2worlds, inv_scale = rescale_poses(cam2worlds)

    # from utils.visual import display_cameras
    # display_cameras(cam2worlds, scale=.1)

    for idx, c2w in zip(cam2worlds_idx, cam2worlds):
        category_frame_annotations[idx].viewpoint.R = tuple(map(tuple, c2w[:3, :3]))
        category_frame_annotations[idx].viewpoint.T = tuple(c2w[:3, -1])
        # transform focal length from NDC to real
        # category_frame_annotations[idx].viewpoint.focal_length = (
        #     category_frame_annotations[idx].image.size[1] * category_frame_annotations[idx].viewpoint.focal_length[
        #         0] / 2,
        #     category_frame_annotations[idx].image.size[0] * category_frame_annotations[idx].viewpoint.focal_length[
        #         1] / 2
        # )
        # transform principal point from NDC to real
        # category_frame_annotations[idx].viewpoint.principal_point = (
        #     category_frame_annotations[idx].image.size[1] * category_frame_annotations[idx].viewpoint.principal_point[
        #         0] / 2,
        #     category_frame_annotations[idx].image.size[0] * category_frame_annotations[idx].viewpoint.principal_point[
        #         1] / 2
        # )

        # category_frame_annotations[idx].viewpoint.principal_point = (
        #     category_frame_annotations[idx].image.size[1] / 2,
        #     category_frame_annotations[idx].image.size[0] / 2
        # )

    for frame_annotation in category_frame_annotations:
        if frame_annotation.sequence_name == sequence_name and frame_annotation.frame_number in train_set:
            imgs_arrays['train'].append(frame_annotation)
        if frame_annotation.sequence_name == sequence_name and frame_annotation.frame_number in val_set:
            imgs_arrays['val'].append(frame_annotation)
        if frame_annotation.sequence_name == sequence_name and frame_annotation.frame_number in test_set:
            imgs_arrays['test'].append(frame_annotation)

    return imgs_arrays, inv_scale, inv_transformation


def _load_renderings(data_dir: str, split: str, resize_factor: float = 1.):
    """Load images from disk."""
    cameras_list, inv_transformation, inv_scale = load_images_annotation(data_dir, split)

    # co3d2opencv_intrinsic = np.asarray(
    #     [[1., -0., 0., 0.],
    #      [0., -1., 0., 0.],
    #      [0., -0., 1., 0.],
    #      [0., -0., 0., 1.]], dtype=np.float32)

    co3d2opencv_intrinsic = np.asarray(
        [[1., -0., 0.],
         [0., -1., 0.],
         [0., -0., 1.]], dtype=np.float32)

    images = []
    camtoworlds = []
    intrinsics = []
    depths = []

    for camera_annotation in cameras_list:
        fname = camera_annotation.image.path
        rgb = iio.imread(fname)
        rgb = downsample(rgb, factor=resize_factor, mode=INTER_LANCZOS4)
        if camera_annotation.mask is not None:
            mask = iio.imread(camera_annotation.mask.path)
            mask = downsample(mask.astype(np.float32) / 255., factor=resize_factor, mode=INTER_LINEAR)
            mask[mask < 0.3] = 0.
            mask = (np.ceil(mask) * 255.).astype(np.uint8)[..., None]
        else:
            mask = np.ones(*(rgb.shape[:-1]), 1, dtype=np.float32)
        rgba = np.concatenate((rgb, mask), axis=-1)
        images.append(rgba)

        # build transformation and intrinsic matrix
        R = np.asarray(camera_annotation.viewpoint.R, dtype=np.float32)
        T = np.asarray(camera_annotation.viewpoint.T, dtype=np.float32)

        # add to list of cameras
        camtoworlds.append(
            np.concatenate(
                (
                    np.concatenate((
                        np.asarray(R, dtype=np.float32),
                        np.asarray(T, dtype=np.float32)[:, None]
                    ), axis=1),
                    np.asarray([[0., 0., 0., 1.]], dtype=np.float32)
                ), axis=0
            )
        )

        fx, fy = camera_annotation.viewpoint.focal_length
        px, py = camera_annotation.viewpoint.principal_point

        K = np.zeros((3, 3), dtype=np.float32)
        K[0, 0] = fx / resize_factor
        K[1, 1] = fy / resize_factor
        K[0, 2] = px / resize_factor
        K[1, 2] = py / resize_factor
        K[2, 2] = 1.0
        K = np.ascontiguousarray(K)  # @ co3d2opencv_intrinsic

        intrinsics.append(K)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    camera_intrinsic = np.stack(intrinsics, axis=0)

    # from utils.visual import display_cameras
    # display_cameras(camtoworlds, scale=.1)

    return images, camtoworlds, camera_intrinsic, inv_transformation, inv_scale


class CO3DDataset(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]

    def __init__(
            self,
            datadir: str,
            split: str = 'train',
            downsample: float = 1.0,
            is_stack=False,
            color_bkgd_aug: str = "white",
            n_test_interpolation: int = 0,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.color_bkgd_aug = color_bkgd_aug
        self.images, self.camtoworlds, _K, inv_transformation, inv_scale = _load_renderings(
            datadir, split, resize_factor=downsample
        )
        self.training = split in ["train", "trainval"]
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.Ks = torch.from_numpy(_K).to(torch.float32)  # (3, 3)
        self.width = self.images.shape[-2]
        self.height = self.images.shape[-3]

        self.scene_bbox = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.white_bg = True
        self.near_far = [0.1, .8]
        self.img_wh = (int(self.images.shape[-2]), int(self.images.shape[-3]))
        self.focal = [self.Ks[0, 0, 0], self.Ks[0, 1, 1]]

        self.n_test_interpolation = n_test_interpolation if split not in ["train", "trainval"] else 0
        self.is_stack = is_stack

        intrinsic_parameters = torch.tensor([self.img_wh[1], self.img_wh[0]])
        _, self.directions = get_rays(intrinsic_parameters, torch.eye(4, dtype=torch.float32)[None], self.Ks[[0]],
                                      intrinsic_parameters.device)

        self.directions = self.directions.reshape(self.img_wh[0] * self.img_wh[1], 3)

        self.generate_rays()

        up = normalize(self.camtoworlds[:, :3, 1].sum(0))
        self.render_path = create_spiral(self.scene_bbox, up, invert_z=False)

    def generate_rays(self):
        self.all_rays = []
        self.all_rgba = []
        self.interpolation = []

        for index in range(len(self)):
            image_id = [index // (self.n_test_interpolation + 1)]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            x_flatten = x.flatten()
            y_flatten = y.flatten()
            x = x[None]
            y = y[None]

            # generate rays
            if index % (self.n_test_interpolation + 1) == 0 or self.split == "train":
                # if self.split != "train":
                # 	print(f"Real image, {index}, idx {image_id[0]}")
                c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
                k = self.Ks[image_id]
                rgba = self.images[image_id, y_flatten, x_flatten] / 255.0  # (num_rays, 3)
                interpolate = False
            else:
                # Interpolation mode
                previous_image = [index // (self.n_test_interpolation + 1)]
                next_image = [(index // (self.n_test_interpolation + 1)) + 1]
                previous_c2w = self.camtoworlds[previous_image]  # (num_rays, 3, 4)
                next_c2w = self.camtoworlds[next_image]  # (num_rays, 3, 4)
                previous_K = self.Ks[previous_image]  # (num_rays, 3, 3)
                next_K = self.Ks[next_image]  # (num_rays, 3, 3)

                alpha = float(index - ((self.n_test_interpolation + 1) * previous_image[0])) / float(
                    ((self.n_test_interpolation + 1) * next_image[0]) - (
                                (self.n_test_interpolation + 1) * previous_image[0]))
                c2w = (next_c2w - previous_c2w) * alpha + previous_c2w
                k = (next_K - previous_K) * alpha + previous_K
                rgba = torch.full_like(self.images[previous_image, y_flatten, x_flatten], 1.0)
                interpolate = True
                # print(f"Generated image, {index}, previous img {previous_image[0]} next img {next_image[0]}, alpha {alpha}")

            origins, directions = get_rays_for_each_pixel(x.to(self.camtoworlds.dtype), y.to(self.camtoworlds.dtype),
                                                          c2w, k)
            viewdirs = directions / torch.linalg.norm(
                directions, dim=-1, keepdims=True
            )

            origins = torch.reshape(origins, (self.height * self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height * self.width, 3))
            rgba = torch.reshape(rgba, (self.height * self.width, 4))

            self.all_rgba += [rgba]
            self.all_rays += [torch.cat([origins, viewdirs], 1)]  # (h*w, 6)
            self.interpolation += [torch.broadcast_to(torch.tensor(interpolate, dtype=torch.bool, device=rgba.device),
                                                      (*(rgba.shape[:-1]), 1))]

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgba = torch.cat(self.all_rgba, 0)  # (len(self.meta['frames])*h*w, 3)
            self.interpolation = torch.cat(self.interpolation, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgba = torch.stack(self.all_rgba, 0).reshape(-1, self.height, self.width,
                                                                  4)  # (len(self.meta['frames]),h,w,3)
            self.interpolation = torch.stack(self.interpolation, 0)  # (len(self.meta['frames]),h*w, 3)

    def __len__(self):
        return len(self.images) + (0 if self.training else self.n_test_interpolation * (len(self.images) - 1))

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "rgbs": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def fetch_data(self, idx):
        """Fetch the data (it maybe cached for multiple batches)."""

        sample = {'rays': self.all_rays[idx],
                  'rgba': self.all_rgba[idx],
                  'interpolation': self.interpolation[idx],
                  'render_path': self.interpolation[idx]}

        return sample
