import json
import os

import numpy as np
from statistics import mean
import pandas as pd


def compute_angular_error(rotation1, rotation2):
    R_rel = rotation1.T @ rotation2
    tr = (np.trace(R_rel) - 1) / 2
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi


def compute_translation_error(translation1, translation2):
    return np.linalg.norm(translation1 - translation2)


def evaluate_poses(filepath: str, csv_filepath):
    with open(filepath, 'r') as fh:
        results = json.load(fh)

    angular_errors = []
    translation_errors = []
    inference_times = []
    sequence_ids = []
    category_names = []
    frame_ids = []
    for result in results:
        gt_c2w = np.eye(4)
        gt_c2w = np.asarray(result['gt_c2w'])
        gt_w2c = np.linalg.inv(gt_c2w)
        pred_c2w = np.asarray(result['pred_c2w'])
        pred_w2c = np.linalg.inv(pred_c2w)
        angular_error = compute_angular_error(gt_w2c[:3, :3], pred_w2c[:3, :3])
        angular_errors.append(angular_error)
        gt_camera_position = np.asarray([0., 0., 0., 1.]).reshape(1, 4) @ gt_c2w[:3, :].T
        pred_camera_position = np.asarray([0., 0., 0., 1.]).reshape(1, 4) @ pred_c2w[:3, :].T
        translation_error = compute_translation_error(gt_camera_position, pred_camera_position)
        translation_errors.append(translation_error)
        inference_times.append(result['total_optimization_time_in_ms'])
        sequence_ids.append(result['sequence_id'])
        category_names.append(result['category_name'])
        frame_ids.append(result['frame_id'])

    df = pd.DataFrame({'category_names': category_names, 'frame_ids': frame_ids, 'sequence_ids': sequence_ids, 'angular_errors': angular_errors, 'translation_errors': translation_errors, 'inference_times': inference_times})
    df.to_csv(csv_filepath)
    mean_angular_error = mean(angular_errors)
    mean_translation_error = mean(translation_errors)
    mean_inference_time = mean(inference_times) / 1000
    print("File: ", filepath)
    print("Mean angular error: ", mean_angular_error)
    print("Mean translation error: ", mean_translation_error)
    print("Mean inference time in s: ", mean_inference_time)

    return mean_angular_error, mean_translation_error, mean_inference_time


if __name__ == "__main__":
    filepath = "/3dv_lego_results/exc_pose_eval_ours.json"
    csv_filepath = '/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/3dv_lego_results/pose_eval_ours.csv'
    mean_angular_error, mean_translation_error, mean_inference_time = evaluate_poses(filepath, csv_filepath)
