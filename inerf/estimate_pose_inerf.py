import time

import numpy as np
import torch
from cv2 import (
    dilate,
    INTER_LANCZOS4,
    INTER_LINEAR,
)
from imageio.v2 import imread
import torch.nn.functional as F

from dataLoader.utils import downsample
from models.tensorBase import TensorBase
from ray_utils import (
    get_ray_directions_Ks,
    get_rays,
)
from inerf.dice_loss import SoftDiceLossV2
from inerf.inerf import find_POI, CameraTransfer, img2mse


def pose_estimation(
    start_pose: torch.Tensor,
    obs_img: np.ndarray,
    cam_K: torch.Tensor,
    model: TensorBase,
    sampling_strategy="interest_regions",
    lrate: float = 0.02,
    optimizer_type: str = "adam",
    batch_size: int = 1024,
    kernel_size: int = 35,
    dil_iter: int = 1,
    color_bkgd_aug: str = "random",
    device: str = "cuda",
    n_iters=1000,
    dice_loss=False,
    print_progress=True,
    target_camera_position=None,
):
    H, W = obs_img.shape[0], obs_img.shape[1]
    # find points of interest of the observed image

    POI = find_POI(
        obs_img[..., :3]
    )  # xy pixel coordinates of points of interest (N x 2)

    # create meshgrid from the observed image
    coords = np.asarray(
        np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
        dtype=int,
    )

    # create sampling mask for interest region sampling strategy
    interest_regions = np.zeros(
        (
            H,
            W,
        ),
        dtype=np.uint8,
    )
    interest_regions[POI[:, 1], POI[:, 0]] = 1
    I = dil_iter
    interest_regions = dilate(
        interest_regions, np.ones((kernel_size, kernel_size), np.uint8), iterations=I
    )
    interest_regions = np.array(interest_regions, dtype=bool)
    interest_regions = coords[interest_regions]

    # not_POI -> contains all points except of POI
    unique_POI = np.unique(POI, axis=0)
    not_POI_mask = np.ones_like(coords[..., 0], dtype=bool)

    not_POI_mask[unique_POI[:, 1], unique_POI[:, 0]] = False
    not_POI = coords[not_POI_mask]

    # coords to a single dimension
    coords = coords.reshape(-1, 2)

    # Create pose transformation model
    cam_transf = CameraTransfer(start_pose).to(device)
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999)
        )
    elif optimizer_type == "adamW":
        optimizer = torch.optim.AdamW(
            params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999)
        )
    else:
        raise ValueError("optimizer type is invalid")

    optimization_poses = []

    soft_dice_loss = SoftDiceLossV2()

    ori_directions, dx, dy = get_ray_directions_Ks(H, W, cam_K, use_pixel_centers=True)
    directions = ori_directions / torch.linalg.norm(
        ori_directions, dim=-1, keepdim=True
    )

    start_time = time.time()
    # imgs - array with images are used to create a video of optimization process
    for k in range(n_iters):
        optimizer.zero_grad()

        if sampling_strategy == "random":
            rand_inds = np.random.choice(
                coords.shape[0], size=batch_size, replace=False
            )
            batch = coords[rand_inds]
        elif sampling_strategy == "interest_points":
            if POI.shape[0] >= batch_size:
                rand_inds = np.random.choice(
                    POI.shape[0], size=batch_size, replace=False
                )
                batch = POI[rand_inds]
            else:
                batch = np.zeros((batch_size, 2), dtype=int)
                batch[: POI.shape[0]] = POI
                rand_inds = np.random.choice(
                    not_POI.shape[0], size=batch_size - POI.shape[0], replace=False
                )
                batch[POI.shape[0] :] = not_POI[rand_inds]
        elif sampling_strategy == "interest_regions":
            rand_inds = np.random.choice(
                interest_regions.shape[0], size=batch_size, replace=False
            )
            batch = interest_regions[rand_inds]
        else:
            print("Unknown sampling strategy")
            return

        target_s = obs_img[batch[:, 1], batch[:, 0]]
        # Change background color
        target_s = torch.from_numpy(target_s).to(device)
        rgb, alpha = torch.split(target_s, [3, 1], dim=-1)

        if color_bkgd_aug == "white":
            color_bkgd = torch.ones((3,), dtype=target_s.dtype, device=target_s.device)
        elif color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, dtype=target_s.dtype, device=target_s.device)
        else:  # args.color_bkgd == 'black'
            color_bkgd = torch.zeros((3,), dtype=target_s.dtype, device=target_s.device)

        target_s = rgb * alpha + color_bkgd * (1.0 - alpha)

        pose = cam_transf()
        rays_o, rays_d, radii = get_rays(
            directions,
            pose,
            directions=ori_directions,
            dx=dx,
            dy=dy,
            keepdim=True,
        )  # both (H, W, 3)

        # Set the color background on observed image
        rays_o = rays_o[0, batch[:, 1], batch[:, 0]]  # (N_rand, 3)
        rays_d = F.normalize(rays_d[0, batch[:, 1], batch[:, 0]], p=2, dim=-1)
        radii = radii[0, batch[:, 1], batch[:, 0]]  # (N_rand, 1)

        # render
        rays_chunk = torch.cat((rays_o, rays_d, radii), dim=-1)

        rgb, _, opacity, _, _, _ = model(
            rays_chunk, bg_color=color_bkgd, is_train=False
        )
        rgb_loss = img2mse(rgb, target_s)
        loss = torch.clone(rgb_loss)
        if dice_loss:
            opacity = torch.clamp(opacity.squeeze(-1), 1.0e-3, 1.0 - 1.0e-3)
            loss += soft_dice_loss(opacity[..., None], alpha)[0]

        create_graph = False
        if optimizer_type == "adahessian":
            create_graph = True
        loss.backward(create_graph=create_graph)
        optimizer.step()

        optimization_poses.append(
            cam_transf().detach().to(device="cpu", non_blocking=True)
        )

        new_lrate = lrate * (0.8 ** ((k + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        if ((k + 1) % 20 == 0 or k == 0) and print_progress:
            print(f"[{k}] Loss: {rgb_loss.item()}")

    if print_progress:
        print(f"Total optimization time: {time.time() - start_time:.02f} s")

    return rgb_loss.item(), cam_transf().detach().cpu(), optimization_poses


def read_image(args):
    img_rgba = np.asarray(imread(args.image_path)).astype(np.float32) / 255.0
    img_rgba = downsample(img_rgba, factor=args.resize_factor, mode=INTER_LANCZOS4)
    if args.mask_path is not None:
        mask = np.asarray(imread(args.mask_path)).astype(np.float32) / 255.0
        mask = np.ceil(downsample(mask, factor=args.resize_factor, mode=INTER_LINEAR))[
            ..., None
        ]
        img_rgba = np.concatenate((img_rgba, mask), axis=-1)
    else:
        mask = np.ones((*(img_rgba.shape[:2]), 1), dtype=img_rgba.dtype)
        img_rgba = np.concatenate((img_rgba, mask), axis=-1)

    return img_rgba
