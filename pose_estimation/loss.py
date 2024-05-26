import torch
from typing import Tuple

def best_one_to_one_rays_selector(
    camera_intrinsic,
    camera_pose,
    obs_img_shape,
    rays_dir,
    rays_ori,
    backbone_wh,
    tanh_denominator=1.0,
):
    gt_camera_position = (
        torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=camera_pose.dtype,
            device=camera_pose.device,
        ).reshape(1, 4)
        @ camera_pose[:3, :].T
    )
    vector_to_point = gt_camera_position - rays_ori
    # batch version of torch.dot(vector_to_point, rays_dir)
    projection_length = torch.bmm(
        vector_to_point.view(vector_to_point.shape[0], 1, vector_to_point.shape[-1]),
        rays_dir.view(rays_dir.shape[0], rays_dir.shape[-1], 1),
    )[
        ..., 0
    ]  # one dim leaved for avoid using none after

    closest_point_along_ray = torch.where(
        projection_length < 0,
        rays_ori,
        rays_ori + torch.multiply(projection_length, rays_dir),
    )
    distance = torch.linalg.norm(closest_point_along_ray - gt_camera_position, dim=-1)
    target_score = 1 - torch.tanh(distance / tanh_denominator)
    point_distance = torch.linalg.norm(vector_to_point, dim=-1)

    point_distance_score = 1 - torch.tanh(point_distance / tanh_denominator)
    target_score_with_distance = torch.multiply(target_score, point_distance_score)

    projection_matrix = camera_intrinsic @ torch.linalg.inv(camera_pose)[:3, :]
    cam_pixels = (
        projection_matrix
        @ torch.cat(
            (
                rays_ori.mT,
                torch.ones(
                    1,
                    rays_ori.shape[0],
                    dtype=rays_ori.dtype,
                    device=rays_dir.device,
                ),
            ),
            dim=0,
        )
    ).mT  # this is valid because the rays_ori was moved to the object surface
    cam_pixels = torch.divide(cam_pixels[..., :2], cam_pixels[..., [-1]])
    # process to convert to feature space coordinates
    backbone_scaling = 256
    if obs_img_shape[0] < obs_img_shape[1]:
        # width is lower
        width_scale_factor = backbone_scaling / obs_img_shape[0]
        height_scale_factor = width_scale_factor
    else:
        # height is lower
        height_scale_factor = backbone_scaling / obs_img_shape[1]
        width_scale_factor = height_scale_factor
    cam_pixels[:, 0] = cam_pixels[:, 0] * width_scale_factor
    cam_pixels[:, 1] = cam_pixels[:, 1] * height_scale_factor
    # center crop
    backbone_crop = 224
    cam_pixels[:, 0] -= ((width_scale_factor * obs_img_shape[0]) - backbone_crop) // 2
    cam_pixels[:, 1] -= ((height_scale_factor * obs_img_shape[1]) - backbone_crop) // 2
    patch_size = 14.0
    cam_pixels = cam_pixels / patch_size
    is_inside = (
        (cam_pixels[..., 1] >= 0.0)
        & (cam_pixels[..., 1] <= backbone_wh[1])
        & (cam_pixels[..., 0] >= 0.0)
        & (cam_pixels[..., 0] <= backbone_wh[0])
    )

    return None, is_inside, target_score, target_score_with_distance


class DistanceBasedScoreLoss(torch.nn.Module):
    def __init__(
        self,
        reweight_method="none",
        lds=False,
        lds_kernel="gaussian",
        lds_ks=5,
        lds_sigma=2,
        total_number_of_elements: float = 256.0,
    ):
        super().__init__()

        assert reweight_method in {"none", "inverse", "sqrt_inv"}
        assert (
            reweight_method != "none" if lds else True
        ), "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"
        self.reweight_method = reweight_method
        self.lds = lds
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma

    def forward(
        self,
        pred_score: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        total_number_of_features: int,
        backbone_wh: Tuple[int, int],
        model_up=None,
        obs_img_shape=(800, 800),
    ):
        with torch.no_grad():
            (
                _,
                _,
                target_score,
                _,
            ) = best_one_to_one_rays_selector(
                camera_intrinsic,
                camera_pose,
                obs_img_shape,
                rays_dir,
                rays_ori,
                backbone_wh=backbone_wh,
                tanh_denominator=1.0,
            )

            combined_score = target_score

            score_multiplier = total_number_of_features / combined_score.sum()

            combined_score = torch.multiply(combined_score, score_multiplier)

        score_diff = torch.square(pred_score - combined_score)

        avg_score = score_diff.mean()
        return avg_score, combined_score
