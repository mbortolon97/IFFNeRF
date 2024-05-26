import numpy as np
import torch
import torch.nn as nn
from cv2 import cvtColor, COLOR_RGB2GRAY, SIFT_create
from kornia.geometry.liegroup import Se3

rot_psi = lambda phi: np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
)

rot_theta = lambda th: np.array(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
)

rot_phi = lambda psi: np.array(
    [
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

trans_t = lambda t: np.array(
    [[1, 0, 0, t[0]], [0, 1, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]]
)


def find_POI(img_rgb):  # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = (cvtColor(img, COLOR_RGB2GRAY) * 255.0).astype(np.uint8)
    sift = SIFT_create()
    keypoints = sift.detect(img_gray, None)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy  # pixel coordinates


def vec2ss_matrix(vector):  # vector to skewsym. matrix
    ss_matrix = torch.zeros((3, 3), dtype=vector.dtype, device=vector.device)
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


class CameraTransfer(nn.Module):
    def __init__(self, start_pose: torch.Tensor):
        super(CameraTransfer, self).__init__()
        self.start_pose = start_pose
        self.w = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0.0, 1e-6, size=()))

    def forward(self):
        exp_i = torch.zeros(
            (4, 4), dtype=self.start_pose.dtype, device=self.start_pose.device
        )
        w_skewsym = vec2ss_matrix(self.w)
        exp_i[:3, :3] = (
            torch.eye(3, dtype=self.start_pose.dtype, device=self.start_pose.device)
            + torch.sin(self.theta) * w_skewsym
            + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        )
        exp_i[:3, 3] = torch.matmul(
            torch.eye(3, dtype=self.start_pose.dtype, device=self.start_pose.device)
            * self.theta
            + (1 - torch.cos(self.theta)) * w_skewsym
            + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym),
            self.v,
        )
        exp_i[3, 3] = 1.0
        T_i = torch.matmul(exp_i, self.start_pose)
        return T_i


# class CameraTransfer(nn.Module):
#     def __init__(self, start_pose: torch.Tensor):
#         super(CameraTransfer, self).__init__()
#
#         self.pose = Se3.from_matrix(start_pose)
#
#     def forward(self):
#         return self.pose


img2mse = lambda x, y: torch.mean((x - y) ** 2)
