import math
from typing import Optional, Tuple

from models.ref import Ref
from models.tensorBase import AlphaGridMask, TensorBase
import torch
from pose_estimation.isocell import isocell_distribution, rotate_isocell


@torch.no_grad()
def jitter_points(samples: torch.Tensor, d):  # [1,n,3]
    # uniform sampling on a sphere
    theta = (
        2
        * math.pi
        * torch.rand(samples.shape[0], dtype=samples.dtype, device=samples.device)
    )
    phi = torch.arccos(
        1 - 2 * torch.rand(samples.shape[0], dtype=samples.dtype, device=samples.device)
    )
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    # rand distance sampling
    distance = torch.abs(
        torch.normal(
            0.0, d, size=(samples.shape[0],), dtype=samples.dtype, device=samples.device
        )
    )
    # combine in a vector
    vector = torch.multiply(torch.stack((x, y, z), dim=-1), distance[:, None])
    return samples + vector


@torch.no_grad()
def multiple_jitter_points(samples: torch.Tensor, d, n_multiple: int = 5):  # [1,n,3]
    # uniform sampling on a sphere
    theta = (
        2
        * math.pi
        * torch.rand(
            samples.shape[0], n_multiple, dtype=samples.dtype, device=samples.device
        )
    )
    phi = torch.arccos(
        1
        - 2
        * torch.rand(
            samples.shape[0], n_multiple, dtype=samples.dtype, device=samples.device
        )
    )
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    # rand distance sampling
    distance = torch.abs(
        torch.normal(
            0.0,
            d,
            size=(samples.shape[0], n_multiple),
            dtype=samples.dtype,
            device=samples.device,
        )
    )
    # combine in a vector
    vector = torch.multiply(torch.stack((x, y, z), dim=-1), distance[..., None])
    return samples[:, None] + vector


def has_valid_occupancy_grid(model: TensorBase):
    if not hasattr(model, "alphaMask"):
        return False
    if not isinstance(model.alphaMask, AlphaGridMask):
        return False
    return True


@torch.no_grad()
def generate_samples_from_occupancy_grid(
    occupancy_grid: AlphaGridMask, num_points: int
):
    grid = occupancy_grid.alpha_volume[0, 0]
    index = torch.stack(
        torch.meshgrid(
            torch.arange(0, grid.shape[0], dtype=torch.float32, device=grid.device),
            torch.arange(0, grid.shape[1], dtype=torch.float32, device=grid.device),
            torch.arange(0, grid.shape[2], dtype=torch.float32, device=grid.device),
            indexing="ij",
        ),
        dim=-1,
    )
    index = index[..., [2, 1, 0]]

    valid_top_back_left = index[grid.to(dtype=torch.bool)]
    idx = torch.randint(
        0,
        valid_top_back_left.shape[0],
        size=(num_points,),
        dtype=torch.long,
        device=valid_top_back_left.device,
    )
    samples = valid_top_back_left[idx]
    samples = samples + torch.rand(
        *samples.shape, dtype=samples.dtype, device=samples.device
    )

    grid_shape = torch.tensor(grid.shape, device=grid.device, dtype=torch.float32)[
        [2, 1, 0]
    ]
    
    xyz_samples = (
        torch.divide(torch.multiply(occupancy_grid.aabbSize, samples), grid_shape - 1.0)
        + occupancy_grid.aabb[0]
    )

    return xyz_samples


def generate_uniform_samples(model: TensorBase, gen_points=5000):
    aabb_size = model.aabb[1] - model.aabb[0]
    samples = (
        torch.multiply(
            torch.rand(gen_points, 3, device=model.aabb.device, dtype=model.aabb.dtype),
            aabb_size,
        )
        + model.aabb[0]
    )
    return samples


@torch.no_grad()
def generate_initial_samples(model: TensorBase, gen_points=5000):
    if has_valid_occupancy_grid(model):
        samples = generate_samples_from_occupancy_grid(model.alphaMask, gen_points)
    else:
        samples = generate_uniform_samples(model, gen_points=gen_points)

    alpha = model.compute_alpha(samples)

    return samples, alpha


@torch.no_grad()
def sampling_epoch(
    samples: torch.Tensor,
    alpha_old: torch.Tensor,
    rho,
    model: TensorBase,
    max_iterations=100,
    n_total_multiple: Optional[int] = None,
):
    # number of point for each input sampling
    if n_total_multiple is None:
        n_total_multiple = samples.shape[0] * 5

    # mask of the non valid sample
    non_valid_sample_masks = torch.ones(
        samples.shape[0], dtype=torch.bool, device=samples.device
    )
    sample_idx = torch.arange(
        non_valid_sample_masks.shape[0], dtype=torch.long, device=samples.device
    )

    thresh_1 = torch.quantile(alpha_old, q=0.6)  # 2 sigmas away
    non_valid_nums = torch.count_nonzero(non_valid_sample_masks)
    iteration = 0
    while non_valid_nums != 0 and iteration < max_iterations:
        n_multiple = n_total_multiple // non_valid_nums
        new_samples = multiple_jitter_points(
            samples[non_valid_sample_masks], rho, n_multiple=n_multiple
        )
        alpha_new = model.compute_alpha(new_samples.view(-1, 3)).view(
            *new_samples.shape[:-1]
        )
        alpha_new_idx_sort = torch.argsort(alpha_new, dim=-1, descending=True)
        indices = torch.argwhere(
            torch.take_along_dim(alpha_new, alpha_new_idx_sort, dim=-1) > thresh_1
        )
        max_index = torch.full(
            alpha_new.shape[:1], -1, dtype=indices.dtype, device=indices.device
        )
        max_index.scatter_reduce_(
            0, indices[:, 0], indices[:, 1], reduce="amax", include_self=False
        )
        new_valid_sample_masks = max_index != -1

        end_shape_size = (torch.count_nonzero(new_valid_sample_masks).item(),)
        select_id = torch.multiply(
            torch.rand(*end_shape_size, dtype=torch.float32, device=samples.device),
            max_index[new_valid_sample_masks] + 0.99,
        ).to(dtype=torch.long)
        original_select_id = torch.take_along_dim(
            alpha_new_idx_sort[new_valid_sample_masks], select_id[:, None], dim=-1
        )
        sorted_alpha = torch.take_along_dim(
            alpha_new[new_valid_sample_masks], original_select_id, dim=-1
        )[:, 0]
        sorted_samples = torch.take_along_dim(
            new_samples[new_valid_sample_masks],
            original_select_id[..., None, :],
            dim=-2,
        )[:, 0]

        it_sample = sample_idx[non_valid_sample_masks][new_valid_sample_masks]

        samples[it_sample] = sorted_samples
        alpha_old[it_sample] = sorted_alpha

        non_valid_sample_masks[it_sample] = False
        non_valid_nums = torch.count_nonzero(non_valid_sample_masks)
        iteration += 1
    
    return samples, alpha_old, iteration, non_valid_nums


def sampling_sphere(
    dtype: torch.dtype = torch.float32, device: torch.device = "cpu", num_viewdirs=5000
):
    sampling = torch.rand(num_viewdirs, 2, dtype=dtype, device=device)
    theta = 2 * math.pi * sampling[..., 0]
    phi = torch.arccos(1 - 2 * sampling[..., 1])
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    return torch.stack((x, y, z), dim=-1)


def sampling_isocell(
    dtype: torch.dtype = torch.float32, device: torch.device = "cpu", num_targets=27
):
    return isocell_distribution(
        num_targets, dtype, device, N0=3, isrand=-1, int_dtype=torch.int64
    )


def evaluate_viewdirs_color(
    point_sampling: torch.Tensor,  # [n, 3]
    viewdir: torch.Tensor,  # [n, 3]
    model: TensorBase,
    **kwargs
):
    point_sampling = torch.broadcast_to(point_sampling, viewdir.shape)

    rgb_map, _, _, _, _, _ = model(
        torch.cat((point_sampling, viewdir), dim=-1).view(-1, 6),
        N_samples=20,
        sample_func=model.sample_point_color,
        **kwargs
    )
    return rgb_map.view(*viewdir.shape)


def bind_viewdirs(
    point_normals: torch.Tensor, sample_dirs: torch.Tensor  # [n, 3]  # [n, 3]
):
    dirs_idx = torch.arange(
        0, sample_dirs.shape[0], dtype=torch.long, device=point_normals.device
    )
    ptx_assign_idx = torch.empty(
        sample_dirs.shape[0], dtype=torch.long, device=point_normals.device
    )
    remaining_complete_dirs = torch.ones(
        sample_dirs.shape[0], dtype=torch.bool, device=point_normals.device
    )
    tot_non_zero = torch.count_nonzero(remaining_complete_dirs).item()
    while tot_non_zero != 0:
        sample_pts_idx = torch.randint(
            low=0,
            high=point_normals.shape[0],
            size=(tot_non_zero,),
            dtype=torch.long,
            device=point_normals.device,
        )
        consider_normals = point_normals[sample_pts_idx]
        consider_dirs = sample_dirs[remaining_complete_dirs]

        # normals are pointing toward the outside while viewing directions are point towards the surface, so they
        # should be opposite. As such if they are pointing in the opposite hemisphere product should be negative
        # exclude equal to 0. as they can generate problems and probability is residual
        dot_product = torch.bmm(
            consider_dirs.view(-1, 1, 3), consider_normals.view(-1, 3, 1)
        )[..., 0, 0]
        valid_combinations = dot_product < -1e-5

        valid_dirs_idx = dirs_idx[remaining_complete_dirs][valid_combinations]
        ptx_assign_idx[valid_dirs_idx] = sample_pts_idx[valid_combinations]
        remaining_complete_dirs[valid_dirs_idx] = False

        tot_non_zero = torch.count_nonzero(remaining_complete_dirs).item()

    return ptx_assign_idx


def unravel_index(
    indices: torch.Tensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(
        indices.size() + shape.size(), dtype=torch.long, device=indices.device
    )

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


@torch.no_grad()
def generate_initial_viewdirs(
    point_sampling: torch.Tensor,  # [n, 3]
    point_normals: torch.Tensor,
    obs_img: torch.Tensor,
    model: TensorBase,
    num_viewdirs=10240,
    target_camera_position=None,
    sample_isocell_targets=27,
    selection_isocell_targets=27,
):
    sample_dirs = sampling_isocell(
        dtype=point_sampling.dtype,
        device=point_sampling.device,
        num_targets=sample_isocell_targets,
    )
    rotated_directions = rotate_isocell(sample_dirs, point_normals)
    rotated_directions = torch.divide(
        rotated_directions, torch.linalg.norm(rotated_directions, dim=-1, keepdim=True)
    )
    # the function is the same but its real use is to display the selected viewdirs
    point_sampling_broadcast = torch.broadcast_to(
        point_sampling[:, None], rotated_directions.shape
    )

    pts_per_chunk = num_viewdirs // sample_dirs.shape[0]
    chunks = torch.split(
        torch.arange(
            0,
            point_sampling_broadcast.shape[0],
            device=point_sampling.device,
            dtype=torch.long,
        ),
        pts_per_chunk,
    )

    rgbs = []
    distances = []
    rgbs_distances = []
    px_positions = []
    obs_img_coords = torch.stack(
        torch.meshgrid(
            torch.arange(
                0, obs_img.shape[-3], dtype=obs_img.dtype, device=obs_img.device
            ),
            torch.arange(
                0, obs_img.shape[-2], dtype=obs_img.dtype, device=obs_img.device
            ),
            indexing="ij",
        ),
        dim=-1,
    )
    for chunk in chunks:
        if target_camera_position is not None:  # TODO: remove debug purpose only
            # Vector from line origin to the point
            target_camera_position = target_camera_position.to(
                device=point_sampling_broadcast.device
            )
            line_to_point = target_camera_position[
                None, :3, -1
            ] - point_sampling_broadcast[chunk].view(-1, 3)

            # Compute the perpendicular distance between the point and the line
            line_direction = rotated_directions[chunk].view(-1, 3)
            min_distances = torch.norm(
                torch.cross(line_to_point, line_direction), dim=-1
            ) / torch.norm(line_direction, dim=-1)
            
            rgb = torch.ones_like(point_sampling_broadcast[chunk].view(-1, 3))
            rgbs.append(rgb.detach().cpu())

            rgb_distance_argmin = torch.zeros_like(
                rgb[:, 0], device=rgb.device, dtype=torch.long
            )
            
            min_rgb_distances = torch.zeros_like(
                rgb[:, 0], device=rgb.device, dtype=torch.float32
            )
            rgbs_distances.append(min_rgb_distances.detach().cpu())
            px_position = torch.index_select(
                obs_img_coords.view(-1, 2), 0, rgb_distance_argmin
            )
            px_positions.append(px_position)
        else:
            line_direction = rotated_directions[chunk].view(-1, 3)
            rgb = evaluate_viewdirs_color(
                point_sampling_broadcast[chunk].view(-1, 3), line_direction, model
            )
            rgbs.append(rgb.detach().cpu())
            distance = torch.cdist(rgb, obs_img.view(-1, 3))
            distance_argmin = torch.argmin(distance, dim=-1)
            min_distances = torch.gather(distance, 1, distance_argmin.view(-1, 1))[:, 0]
            rgbs_distances.append(min_distances.detach().cpu())
            px_position = torch.index_select(
                obs_img_coords.view(-1, 2), 0, distance_argmin
            )
            px_positions.append(px_position)
        distances.append(
            min_distances.view(chunk.shape[0], rotated_directions.shape[1])
        )
    distances = torch.cat(distances, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    rgbs_distances = torch.cat(rgbs_distances, dim=0)
    px_positions = torch.cat(px_positions, dim=0)

    top_indices = torch.argsort(distances.reshape(-1), dim=0, descending=False)
    
    return (
        point_sampling_broadcast.reshape(-1, 3)[top_indices],
        rotated_directions.reshape(-1, 3)[top_indices],
        distances.reshape(-1)[top_indices],
        rgbs.reshape(-1, 3)[top_indices.cpu()],
        rgbs_distances.reshape(-1)[top_indices.cpu()],
        px_positions.reshape(-1, 2)[top_indices],
    )


def generate_all_possible_rays(
    point_sampling: torch.Tensor,  # [n, 3]
    point_normals: torch.Tensor,
    model: TensorBase,
    num_viewdirs_per_chunk=10240,
    sample_isocell_targets=27,
):
    sample_dirs = sampling_isocell(
        dtype=point_sampling.dtype,
        device=point_sampling.device,
        num_targets=sample_isocell_targets,
    )
    rotated_directions = rotate_isocell(sample_dirs, point_normals)
    rotated_directions = torch.divide(
        rotated_directions, torch.linalg.norm(rotated_directions, dim=-1, keepdim=True)
    )
    # the function is the same but its real use is to display the selected viewdirs
    point_sampling_broadcast = torch.broadcast_to(
        point_sampling[:, None], rotated_directions.shape
    )

    pts_per_chunk = num_viewdirs_per_chunk // sample_dirs.shape[0]
    chunks = torch.split(
        torch.arange(
            0,
            point_sampling_broadcast.shape[0],
            device=point_sampling.device,
            dtype=torch.long,
        ),
        pts_per_chunk,
    )

    rgbs = []

    for chunk in chunks:
        line_direction = rotated_directions[chunk].view(-1, 3)
        rgb = evaluate_viewdirs_color(
            point_sampling_broadcast[chunk].view(-1, 3), line_direction, model
        )
        rgbs.append(rgb)
    rgbs = torch.cat(rgbs, dim=0)

    return (
        point_sampling_broadcast.reshape(-1, 3),
        rotated_directions.reshape(-1, 3),
        rgbs.reshape(-1, 3),
    )


def compute_consensus(
    point_sampling: torch.Tensor,  # [n, 3]
    viewdir: torch.Tensor,
    min_distance: torch.Tensor,
):
    pass


def viewdir_sampling(
    point_sampling: torch.Tensor,  # [n, 3]
    viewdir: torch.Tensor,
    obs_img: torch.Tensor,  # [n, 3]
    model: TensorBase,  # [n, 3]
    num_viewdirs=70,
):
    pass


def iterative_surface_sampling_process(
    model, gen_points=8000, n_iteration=4, max_resampling_iterations=200
):
    sample_history = [
        generate_uniform_samples(model, gen_points=gen_points).cpu().numpy()
    ]

    samples, alpha = generate_initial_samples(model, gen_points)

    if has_valid_occupancy_grid(model):
        rho = (torch.max(model.gridSize) * 0.1) * torch.max(
            model.aabbSize / model.gridSize
        )
    else:
        rho = torch.linalg.norm(model.aabbSize)
    sample_history.append(samples.cpu().numpy())

    for i in range(n_iteration):
        samples, alpha, iteration, old_data = sampling_epoch(
            samples, alpha, rho, model, max_iterations=max_resampling_iterations
        )
        sample_history.append(samples.cpu().numpy())
    
    return samples


def samples_points_normals(model, samples):
    assert isinstance(
        model.renderModule, Ref
    ), "render module should be able to compute the normal"
    norm_samples = model.normalize_coord(samples)
    app_features = model.compute_appfeature(norm_samples)
    return model.renderModule.compute_normals(app_features)
