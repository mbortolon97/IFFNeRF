"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import numpy as np
from cv2 import resize, INTER_AREA

def downsample(img, factor, patch_size=-1, mode=INTER_AREA):
    """Area downsample img (factor must evenly divide img height and width)."""
    sh = img.shape
    def max_fn(x):
        return int(max(x, patch_size))
    out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
    img = resize(img, out_shape, mode)
    return img

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = np.mean(poses[:, :3, 3], axis=0)
    z_axis = np.mean(poses[:, :3, 2], axis=0)
    up = np.mean(poses[:, :3, 1], axis=0)
    cam2world = viewmatrix(z_axis, up, position)
    cam2world = np.concatenate((cam2world, np.asarray([[0., 0., 0., 1.]], dtype=poses.dtype)), axis=-2)
    return cam2world

"""
Adapted from code originally written by David Novotny.
"""
import torch
from pytorch3d.transforms import Rotate, Translate


def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect = intersect_skew_lines_high_dim(p, r, mask=mask)
    return p_intersect


def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = np.ones_like(p[..., 0])
    r = r / np.linalg.norm(r, axis=-1, keepdims=True)

    I_min_cov = (np.eye(dim, dtype=p.dtype)[None] - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = np.matmul(I_min_cov, p[..., None]).sum(axis=-3)
    p_intersect = np.linalg.lstsq(I_min_cov.sum(axis=-3), sum_proj, rcond=None)[0][..., 0]

    if np.any(np.isnan(p_intersect)):
        print(p_intersect)
        assert False
    return p_intersect


def compute_optical_axis_intersection(cam2world):
    dirs = np.broadcast_to(np.asarray([[[0., 0., 1.]]], dtype=cam2world.dtype), (cam2world.shape[0], 1, 3))
    oa_ray_dir = (dirs @ np.transpose(cam2world[..., :3, :3], axes=(0, 2, 1)))[:, 0]
    oa_ray_ori = cam2world[:, :3, -1]

    # optical_axis = np.asarray([[[0., 0., 1.]]], dtype=np.float32)
    #
    # oa_ray_dir = (cam2world @ np.concatenate([optical_axis.transpose(0, 2, 1), np.ones((1, 1, 1))], axis=-2)).transpose(
    #     0, 2, 1)
    # oa_ray_dir = oa_ray_dir[..., 0, :3] / oa_ray_dir[..., 0, [-1]]
    # oa_ray_ori = cam2world[:, :3, -1]
    #
    p_intersect = intersect_skew_line_groups(
        p=oa_ray_ori, r=oa_ray_dir, mask=None
    )

    # from utils.visual import display_rays_intersection
    # display_rays_intersection(cam2world, oa_ray_ori, oa_ray_dir, p_intersect, scale=2.)

    return p_intersect


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.asarray([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=vec1.dtype)
    rotation_matrix = np.eye(3, dtype=vec1.dtype) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def fit_3D_plane(pts):
    (rows, cols) = pts.shape
    G = np.ones((rows, 3))
    G[:, 0] = pts[:, 0]  # X
    G[:, 1] = pts[:, 1]  # Y
    Z = pts[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return c, normal


def recenter_poses(cam2world, pose_avg=None, method='fitting'):
    """Recenter poses computing the origin using the rays"""
    if pose_avg is not None:
        cam2world_avg = np.linalg.inv(pose_avg)
    elif method == 'pca':
        t = cam2world[:, :3, 3]
        t_mean = t.mean(axis=0)
        t = t - t_mean

        eigval, eigvec = np.linalg.eig(t.T @ t)
        # Sort eigenvectors in order of largest to smallest eigenvalue.
        inds = np.argsort(eigval)[::-1]
        eigvec = eigvec[:, inds]
        rot = eigvec.T
        if np.linalg.det(rot) < 0:
            rot = np.diag(np.array([1, 1, -1])) @ rot

        transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
        poses_recentered = transform @ cam2world
        transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

        # Flip coordinate system if z component of y-axis is negative
        if poses_recentered.mean(axis=0)[2, 1] < 0:
            transform = np.diag(np.array([1, -1, -1, 1])) @ transform

        cam2world_avg = transform
    else:
        cam2world_clone = cam2world.copy()
        # compute the rotation component
        rotation_cam2world = np.eye(4, dtype=cam2world_clone.dtype)
        destination_vector = np.asarray([0., 0., 1.], dtype=np.float32)

        _, plane_normal = fit_3D_plane(cam2world_clone[:, :3, -1])
        rotation_cam2world[:3, :3] = rotation_matrix_from_vectors(plane_normal, destination_vector)
        # compute the translation component
        translation_cam2world = np.eye(4, dtype=cam2world_clone.dtype)
        # rotation_cam2world[None] @
        p_intersect = compute_optical_axis_intersection(rotation_cam2world[None] @ cam2world_clone)
        translation_cam2world[:3, -1] = -p_intersect
        # compose
        cam2world_avg = translation_cam2world @ rotation_cam2world

    cam2world = cam2world_avg[None] @ cam2world
    return cam2world, np.linalg.inv(cam2world_avg)


def recenter_poses_avg(cam2world, pose_avg=None):
    """Recenter poses around the origin."""
    if pose_avg is not None:
        cam2world_avg = pose_avg
    else:
        cam2world_avg = poses_avg(cam2world)[:3, -1]
    cam2world = cam2world_avg @ cam2world
    return cam2world, cam2world_avg


def rescale_poses(poses, scale=None):
    """Rescales camera poses according to maximum x/y/z value."""
    if scale is not None:
        s = scale
    else:
        s = np.max(np.linalg.norm(poses[:, :3, -1], axis=-1))
    poses[:, :3, -1] /= s
    return poses, s


def rescale_poses_max(poses, scale=None):
    """Rescales camera poses according to maximum x/y/z value."""
    if scale is not None:
        s = scale
    else:
        s = np.max(np.abs(poses[:, :3, -1]))
    out = np.copy(poses)
    out[:, :3, -1] /= s
    return out, s
