import json
import os

import numpy as np
from statistics import mean, stdev
import pandas as pd
from typing import List, Optional
from scipy.stats import t
from math import sqrt


def compute_angular_error(rotation_gt, rotation_est):
    cos_angle = (np.trace(rotation_gt @ np.linalg.inv(rotation_est)) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # R_rel = rotation1.T @ rotation2
    # tr = (np.trace(R_rel) - 1) / 2
    # theta = np.arccos(tr.clip(-1, 1))
    # return theta * 180 / np.pi


def compute_translation_error(translation1, translation2):
    return np.linalg.norm(translation1 - translation2)


def compute_confidence_interval(length_data, data_mean, data_std):
    dof = length_data - 1
    confidence = 0.95
    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
    return data_mean - data_std * t_crit / sqrt(length_data), data_mean + data_std * t_crit / sqrt(length_data)


def evaluate_poses(filepath: str,
                   angular_acc_levels: Optional[List[str]] = None, translation_acc_levels: Optional[List[str]] = None):
    with open(filepath, 'r') as fh:
        results = json.load(fh)
    
    if angular_acc_levels is None:
        angular_acc_levels = ['2.5','5.','10.','30.']
    if translation_acc_levels is None:
        translation_acc_levels = ['.1','.2','.5']
    
    translation_acc = {translation_acc_level: 0 for translation_acc_level in translation_acc_levels}
    angular_acc = {angular_acc_level: 0 for angular_acc_level in angular_acc_levels}
    translation_acc_inside_err = {translation_acc_level: [] for translation_acc_level in translation_acc_levels}
    angular_acc_inside_err = {angular_acc_level: [] for angular_acc_level in angular_acc_levels}
    angular_errors = []
    translation_errors = []
    inference_times = []
    photometric_errors = []
    
    for result in results:
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

        for translation_acc_level in translation_acc_levels:
            if float(translation_acc_level) > translation_error:
                translation_acc[translation_acc_level] += 1
                translation_acc_inside_err[translation_acc_level].append(translation_error)
        for angular_acc_level in angular_acc_levels:
            if float(angular_acc_level) > angular_error:
                angular_acc[angular_acc_level] += 1
                angular_acc_inside_err[angular_acc_level].append(angular_error)

        inference_times.append(result['total_optimization_time_in_ms'])
        photometric_errors.append(result['loss'])

    mean_angular_error = mean(angular_errors)
    std_angular_error = stdev(angular_errors)
    angular_conf_int = compute_confidence_interval(len(angular_errors), mean_angular_error, std_angular_error)

    mean_translation_error = mean(translation_errors)
    std_translation_error = stdev(translation_errors)
    translation_conf_int = compute_confidence_interval(
        len(translation_errors), mean_translation_error, std_translation_error)

    mean_inference_time = mean(inference_times) / 1000
    mean_photometric_error = mean(photometric_errors)
    print("Mean angular error: ", mean_angular_error)
    print("Mean translation error: ", mean_translation_error)
    print("Mean inference time in s: ", mean_inference_time)

    angular_acc_ratio = {
        angular_acc_level: (angular_acc[angular_acc_level] / len(results)) * 100.
        for angular_acc_level in angular_acc_levels
    }
    translation_acc_ratio = {
        translation_acc_level: (translation_acc[translation_acc_level] / len(results)) * 100.
        for translation_acc_level in translation_acc_levels
    }

    angular_acc_error = {
        angular_acc_level: mean(angular_acc_inside_err[angular_acc_level]) if len(angular_acc_inside_err[angular_acc_level]) != 0 else np.nan
        for angular_acc_level in angular_acc_levels
    }
    translation_acc_error = {
        translation_acc_level: mean(translation_acc_inside_err[translation_acc_level]) if len(translation_acc_inside_err[translation_acc_level]) != 0 else np.nan
        for translation_acc_level in translation_acc_levels
    }

    return (mean_angular_error, std_angular_error, angular_conf_int, mean_translation_error, std_translation_error,
            translation_conf_int, mean_inference_time, mean_photometric_error, angular_acc_ratio,
            angular_acc_error, translation_acc_ratio, translation_acc_error)


def explore_and_compute_stats(dir_to_explore, out_path):
    # mean_angular_error, mean_translation_error, mean_inference_time = evaluate_poses(os.path.join(dir_to_explore, ""))
    angular_acc_levels = ['2.5', '5.', '10.', '30.']
    translation_acc_levels = ['.1', '.2', '.5']

    df_result = {
        'filename': [],
        'mean_angular_error': [],
        'std_angular_error': [],
        'low_angular_conf_int': [],
        'high_angular_conf_int': [],
        'mean_translation_error': [],
        'std_translation_error': [],
        'low_translation_conf_int': [],
        'high_translation_conf_int': [],
        'mean_photometric_error': [],
        'mean_inference_time': []
    }

    for angular_acc_level in angular_acc_levels:
        df_result[f'acc_{angular_acc_level}_deg'] = []
        df_result[f'err_inside_{angular_acc_level}_deg'] = []
    for translation_acc_level in translation_acc_levels:
        df_result[f'acc_{translation_acc_level}'] = []
        df_result[f'err_inside_{translation_acc_level}'] = []

    files = os.listdir(dir_to_explore)
    for file in files:
        filepath = os.path.join(dir_to_explore, file)
        if not file.startswith("pose_eval_"):
            continue
        if not file.endswith(".json"):
            continue

        mean_angular_error, std_angular_error, angular_conf_int, mean_translation_error, std_translation_error,\
            translation_conf_int, mean_inference_time, mean_photometric_error, angular_acc_ratio, angular_acc_error,\
            translation_acc_ratio, translation_acc_error = evaluate_poses(
                filepath, angular_acc_levels=angular_acc_levels, translation_acc_levels=translation_acc_levels)

        df_result['filename'].append(file)
        df_result['mean_angular_error'].append(mean_angular_error)
        df_result['std_angular_error'].append(std_angular_error)
        df_result['low_angular_conf_int'].append(angular_conf_int[0])
        df_result['high_angular_conf_int'].append(angular_conf_int[1])
        df_result['mean_translation_error'].append(mean_translation_error)
        df_result['std_translation_error'].append(std_translation_error)
        df_result['low_translation_conf_int'].append(translation_conf_int[0])
        df_result['high_translation_conf_int'].append(translation_conf_int[1])
        df_result['mean_photometric_error'].append(mean_photometric_error)
        df_result['mean_inference_time'].append(mean_inference_time)
        for angular_acc_level in angular_acc_levels:
            df_result[f'acc_{angular_acc_level}_deg'].append(angular_acc_ratio[angular_acc_level])
            df_result[f'err_inside_{angular_acc_level}_deg'].append(angular_acc_error[angular_acc_level])
        for translation_acc_level in translation_acc_levels:
            df_result[f'acc_{translation_acc_level}'].append(translation_acc_ratio[translation_acc_level])
            df_result[f'err_inside_{translation_acc_level}'].append(translation_acc_error[translation_acc_level])

    df = pd.DataFrame.from_dict(df_result)

    df = df.sort_values(by=['mean_angular_error', 'mean_translation_error'])

    df.to_excel(out_path)


if __name__ == "__main__":
    explore_and_compute_stats("/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/icra_tt_results",
                              "/home/mbortolon/Downloads/results_icra_tt.xlsx")

