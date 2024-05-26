import os

from compute_eval_statistics import (
    compute_angular_error,
    compute_translation_error,
    compute_confidence_interval,
)
import json

import numpy as np
from statistics import mean, stdev, median
from typing import List, Optional
import pandas as pd


def evaluate_poses_per_object(
    filepath: str, object_id_fields: Optional[List[str]] = None
):
    print(filepath)
    if object_id_fields is None:
        object_id_fields = ["sequence_id", "category_name"]
    with open(filepath, "r") as fh:
        results = json.load(fh)

    results_object_id = [
        "_".join([result[object_id_field] for object_id_field in object_id_fields])
        for result in results
    ]
    objects = set(results_object_id)

    angular_errors = {object_id: [] for object_id in objects}
    translation_errors = {object_id: [] for object_id in objects}
    inference_times = {object_id: [] for object_id in objects}
    photometric_errors = {object_id: [] for object_id in objects}

    for result_object_id, result in zip(results_object_id, results):
        gt_c2w = np.asarray(result["gt_c2w"])
        gt_w2c = np.linalg.inv(gt_c2w)
        pred_c2w = np.asarray(result["pred_c2w"])
        pred_w2c = np.linalg.inv(pred_c2w)
        angular_error = compute_angular_error(gt_w2c[:3, :3], pred_w2c[:3, :3])
        angular_errors[result_object_id].append(angular_error)

        gt_camera_position = (
            np.asarray([0.0, 0.0, 0.0, 1.0]).reshape(1, 4) @ gt_c2w[:3, :].T
        )
        pred_camera_position = (
            np.asarray([0.0, 0.0, 0.0, 1.0]).reshape(1, 4) @ pred_c2w[:3, :].T
        )
        translation_error = compute_translation_error(
            gt_camera_position, pred_camera_position
        )
        translation_errors[result_object_id].append(translation_error)

        inference_times[result_object_id].append(
            result["total_optimization_time_in_ms"]
        )
        photometric_errors[result_object_id].append(result["loss"])

    mean_angular_error = {
        object_id: mean(angular_errors[object_id]) for object_id in objects
    }
    std_angular_error = {
        object_id: stdev(angular_errors[object_id]) for object_id in objects
    }
    median_angular_error = {
        object_id: median(angular_errors[object_id]) for object_id in objects
    }

    angular_conf_int = {
        object_id: compute_confidence_interval(
            len(angular_errors[object_id]),
            mean_angular_error[object_id],
            std_angular_error[object_id],
        )
        for object_id in objects
    }

    mean_translation_error = {
        object_id: mean(translation_errors[object_id]) for object_id in objects
    }
    std_translation_error = {
        object_id: stdev(translation_errors[object_id]) for object_id in objects
    }
    median_translation_error = {
        object_id: median(translation_errors[object_id]) for object_id in objects
    }
    translation_conf_int = {
        object_id: compute_confidence_interval(
            len(translation_errors[object_id]),
            mean_translation_error[object_id],
            std_translation_error[object_id],
        )
        for object_id in objects
    }

    mean_inference_time = {
        object_id: mean(inference_times[object_id]) / 1000 for object_id in objects
    }
    mean_photometric_error = {
        object_id: mean(photometric_errors[object_id]) for object_id in objects
    }

    return (
        mean_angular_error,
        std_angular_error,
        median_angular_error,
        angular_conf_int,
        mean_translation_error,
        std_translation_error,
        median_translation_error,
        translation_conf_int,
        mean_inference_time,
        mean_photometric_error,
    )


def explore_and_compute_stats(dir_to_explore, out_path):
    df_result = {
        "filename": [],
    }

    object_fields = [
        "mean_angular_error",
        "std_angular_error",
        "median_angular_error",
        "low_angular_conf_int",
        "high_angular_conf_int",
        "mean_translation_error",
        "std_translation_error",
        "median_translation_error",
        "low_translation_conf_int",
        "high_translation_conf_int",
        "mean_photometric_error",
        "mean_inference_time",
    ]

    objects = set()

    files = os.listdir(dir_to_explore)
    for file in files:
        filepath = os.path.join(dir_to_explore, file)
        if not file.startswith("pose_eval_"):
            continue
        if not file.endswith(".json"):
            continue

        (
            mean_angular_error,
            std_angular_error,
            median_angular_error,
            angular_conf_int,
            mean_translation_error,
            std_translation_error,
            median_translation_error,
            translation_conf_int,
            mean_inference_time,
            mean_photometric_error,
        ) = evaluate_poses_per_object(filepath, object_id_fields=["sequence_id"])
        
        df_result["filename"].append(file)

        objects_to_fill = objects - set(mean_angular_error.keys())
        for object_to_fill in objects_to_fill:
            for object_field in object_fields:
                df_result[f"{object_field}_{object_to_fill}"].append(np.nan)

        for object_id in mean_angular_error.keys():
            if object_id not in objects:
                objects.add(object_id)

                for object_field in object_fields:
                    df_result[f"{object_field}_{object_id}"] = [
                        np.nan for _ in range(len(df_result["filename"]) - 1)
                    ]

        for object_id, object_mean_angular_error in mean_angular_error.items():
            df_result[f"mean_angular_error_{object_id}"].append(
                object_mean_angular_error
            )
        for object_id, object_std_angular_error in std_angular_error.items():
            df_result[f"std_angular_error_{object_id}"].append(object_std_angular_error)
        for object_id, object_median_angular_error in median_angular_error.items():
            df_result[f"median_angular_error_{object_id}"].append(
                object_median_angular_error
            )
        for object_id, object_angular_conf_int in angular_conf_int.items():
            df_result[f"low_angular_conf_int_{object_id}"].append(
                object_angular_conf_int[0]
            )
            df_result[f"high_angular_conf_int_{object_id}"].append(
                object_angular_conf_int[1]
            )
        for object_id, object_mean_translation_error in mean_translation_error.items():
            df_result[f"mean_translation_error_{object_id}"].append(
                object_mean_translation_error
            )
        for object_id, object_std_translation_error in std_translation_error.items():
            df_result[f"std_translation_error_{object_id}"].append(
                object_std_translation_error
            )
        for (
            object_id,
            object_median_translation_error,
        ) in median_translation_error.items():
            df_result[f"median_translation_error_{object_id}"].append(
                object_median_translation_error
            )
        for object_id, object_translation_conf_int in translation_conf_int.items():
            df_result[f"low_translation_conf_int_{object_id}"].append(
                object_translation_conf_int[0]
            )
            df_result[f"high_translation_conf_int_{object_id}"].append(
                object_translation_conf_int[1]
            )
        for object_id, object_mean_photometric_error in mean_photometric_error.items():
            df_result[f"mean_photometric_error_{object_id}"].append(
                object_mean_photometric_error
            )
        for object_id, object_mean_inference_time in mean_inference_time.items():
            df_result[f"mean_inference_time_{object_id}"].append(
                object_mean_inference_time
            )

    to_discard_fields = [
        "std_angular_error",
        "low_angular_conf_int",
        "high_angular_conf_int",
        "std_translation_error",
        "low_translation_conf_int",
        "high_translation_conf_int",
        "mean_photometric_error",
        "mean_inference_time",
    ]

    to_discard_elements = []
    for df_results_element in df_result.keys():
        for to_discard_field in to_discard_fields:
            if df_results_element.startswith(to_discard_field):
                to_discard_elements.append(df_results_element)

    for to_discard_element in to_discard_elements:
        del df_result[to_discard_element]
    
    breakpoint()

    # print(len(df_result['mean_angular_error_ship']))
    # print(df_result.keys())
    # print({df_result_key: len(df_result_value) for df_result_key, df_result_value in df_result.items()})
    df = pd.DataFrame.from_dict(df_result)

    df.to_excel(out_path)


if __name__ == "__main__":
    explore_and_compute_stats(
        # "/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/3dv_lego_results",
        # "/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/cvpr_mip_results",
        # "/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/cvpr_blender_pinerf_results",
        "/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/icra_tt_results",
        # "/home/mbortolon/Downloads/results_cvpr_synthetic_per_object.xlsx",
        # "/home/mbortolon/Downloads/results_cvpr_mip_per_object.xlsx",
        # "/home/mbortolon/Downloads/results_cvpr_blender_pinerf_per_object.xlsx",
        "/home/mbortolon/Downloads/results_cvpr_tt_per_object.xlsx",
    )
