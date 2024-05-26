"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import os

import imageio.v2 as imageio
import numpy as np
import torch

from .utils import downsample
from .repair_camera_parser import load_cameras_xml
from .rays import get_rays_for_each_pixel, get_rays
from .llff import normalize

from cv2 import INTER_LANCZOS4, INTER_LINEAR
from .spiral_utils import create_spiral

VAL_SPLIT_EVERY = 10

# how to draw a spiral?

def _load_renderings(data_dir: str, split : str, resize_factor : float = 1.):
    """Load images from disk."""
    cameras_xml_path = os.path.join(data_dir, "cameras.xml")
    cameras_dict = load_cameras_xml(cameras_xml_path, data_dir, img_resize_factor=resize_factor)

    if split == "test":
        cameras_dict = {k: v[::VAL_SPLIT_EVERY] for k, v in cameras_dict.items()}
        # print("val frames len: ", len(frames))
    elif split == "train":
        val_idx_list = range(len(cameras_dict['filenames']))[::VAL_SPLIT_EVERY]
        cameras_dict = {k: [in_v for idx, in_v in enumerate(v) if idx not in val_idx_list] for k, v in cameras_dict.items()}
        # print("train frames len: ", len(frames))
    
    images = []
    for i in range(len(cameras_dict['filenames'])):
        fname = cameras_dict['filenames'][i]
        rgb = imageio.imread(fname)
        rgb = downsample(rgb, factor=resize_factor, mode=INTER_LANCZOS4)
        basename = os.path.basename(fname)
        mask_path = os.path.join(data_dir, 'masks', basename)
        mask = imageio.imread(mask_path)
        mask = downsample(mask.astype(np.float32) / 255., factor=resize_factor, mode=INTER_LINEAR)
        mask = (np.ceil(mask) * 255.).astype(np.uint8)[..., None]
        rgba = np.concatenate((rgb, mask), axis=-1)
        images.append(rgba)
    
    images = np.stack(images, axis=0)
    camtoworlds = np.asarray(cameras_dict['cam2world'], dtype=np.float32)
    camera_intrinsic = np.asarray(cameras_dict['Ks'], dtype=np.float32)
    
    return images, camtoworlds, camera_intrinsic

class RepairDataset(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    
    def __init__(
        self,
        datadir: str,
        split: str = 'train',
        downsample: float = 1.0,
        is_stack=False,
        color_bkgd_aug: str = "random",
        n_test_interpolation : int = 0,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.color_bkgd_aug = color_bkgd_aug
        self.images, self.camtoworlds, _K = _load_renderings(
            datadir, split, resize_factor=downsample
        )
        self.training = split in ["train", "trainval"]
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.Ks = torch.from_numpy(_K).to(torch.float32)  # (3, 3)
        self.width = self.images.shape[-2]
        self.height = self.images.shape[-3]

        self.scene_bbox = torch.tensor([[-1.0,-1.0,0.0], [1.0,1.0,1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.white_bg = True
        self.near_far = [0.1,1.8]
        self.img_wh = (int(self.images.shape[-2]),int(self.images.shape[-3]))
        self.focal = [self.Ks[0, 0, 0], self.Ks[0, 1, 1]]

        self.n_test_interpolation = n_test_interpolation if split not in ["train", "trainval"] else 0
        self.is_stack = is_stack

        intrinsic_parameters = torch.tensor([self.img_wh[1], self.img_wh[0]])
        _, self.directions = get_rays(intrinsic_parameters, torch.eye(4, dtype=torch.float32)[None], self.Ks[[0]], intrinsic_parameters.device)
        
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

                alpha = float(index - ((self.n_test_interpolation + 1) * previous_image[0])) / float(((self.n_test_interpolation + 1) * next_image[0]) - ((self.n_test_interpolation + 1) * previous_image[0]))
                c2w = (next_c2w - previous_c2w) * alpha + previous_c2w
                k = (next_K - previous_K) * alpha + previous_K
                rgba = torch.full_like(self.images[previous_image, y_flatten, x_flatten], 1.0)
                interpolate = True
                # print(f"Generated image, {index}, previous img {previous_image[0]} next img {next_image[0]}, alpha {alpha}")
            
            origins, directions = get_rays_for_each_pixel(x.to(self.camtoworlds.dtype), y.to(self.camtoworlds.dtype), c2w, k)
            viewdirs = directions / torch.linalg.norm(
                directions, dim=-1, keepdims=True
            )

            origins = torch.reshape(origins, (self.height * self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height * self.width, 3))
            rgba = torch.reshape(rgba, (self.height * self.width, 4))

            self.all_rgba += [rgba]
            self.all_rays += [torch.cat([origins, viewdirs], 1)]  # (h*w, 6)
            self.interpolation += [torch.broadcast_to(torch.tensor(interpolate, dtype=torch.bool, device=rgba.device), (*(rgba.shape[:-1]), 1))]

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgba = torch.cat(self.all_rgba, 0)  # (len(self.meta['frames])*h*w, 3)
            self.interpolation = torch.cat(self.interpolation, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgba = torch.stack(self.all_rgba, 0).reshape(-1,self.height,self.width,4)  # (len(self.meta['frames]),h,w,3)
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
