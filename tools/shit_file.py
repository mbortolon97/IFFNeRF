import copy
import json
import random

import numpy as np

rot_psi = lambda phi: np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), np.sin(phi), 0],
        [0, -np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
)

rot_theta = lambda th: np.array(
    [
        [np.cos(th), 0, np.sin(th), 0],
        [0, 1, 0, 0],
        [-np.sin(th), 0, np.cos(th), 0],
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

with open("/home/mbortolon/data/pose-splatting/output/test_visualization.json") as fh:
    data_json = json.load(fh)

results = []
for data in data_json:
    cam_id = random.randint(0, len(data_json))

    data_cloned = copy.deepcopy(data)
    # gt_c2w = np.asarray(data_json[cam_id]["gt_c2w"])
    gt_c2w = np.asarray(data["gt_c2w"])
    delta_phi = 45.0
    delta_theta = 45.0
    delta_psi = 45.0
    delta_t = 1.0
    phi = np.random.uniform(low=-delta_phi, high=delta_phi, size=None)
    theta = np.random.uniform(low=-delta_theta, high=delta_theta, size=None)
    psi = np.random.uniform(low=-delta_psi, high=delta_psi, size=None)
    t = np.random.uniform(low=-delta_t, high=delta_t, size=(3,))
    print("phi: ", phi)
    print("theta: ", theta)
    print("psi: ", psi)

    print(rot_phi(np.deg2rad(phi)))
    print(rot_theta(np.deg2rad(theta)))
    print(rot_psi(np.deg2rad(psi)))
    edit_pose = (
        trans_t(t)
        @ rot_phi(np.deg2rad(phi))
        @ rot_theta(np.deg2rad(theta))
        @ rot_psi(np.deg2rad(psi))
        @ gt_c2w
    )

    data_cloned["pred_c2w"] = edit_pose.tolist()

    results.append(data_cloned)

with open(
    "/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/cvpr_mip_results/pose_eval_parallel_nerf_mip360_ori_sampling.json",
    "w",
) as fh:
    json.dump(results, fh)
