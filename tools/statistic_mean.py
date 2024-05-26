import json
import statistics


def combine_pose_eval_files(input_result_file, output_result_file):
    with open(input_result_file, "r") as fh:
        input_results = json.load(fh)

    output_results = {}
    for input_result in input_results:
        key = (
            input_result["sequence_id"],
            input_result["category_name"],
            input_result["frame_id"],
        )
        if key not in output_results:
            output_results[key] = {
                object_key: (
                    value
                    if object_key in ["sequence_id", "category_name", "frame_id"]
                    else [value]
                )
                for object_key, value in input_result.items()
            }
        else:
            for object_key, value in input_result.items():
                if object_key in ["sequence_id", "category_name", "frame_id"]:
                    continue
                output_results[key][object_key].append(value)

    breakpoint()

    output = []
    for input_result in output_results.values():
        output.append(
            {
                object_key: (
                    value
                    if object_key in ["sequence_id", "category_name", "frame_id"]
                    else statistics.mean(value)
                )
                for object_key, value in input_result.items()
            }
        )

    print(output)


if __name__ == "__main__":
    combine_pose_eval_files(
        "/cvpr_blender_pinerf_results/pose_eval_inerf_pinerf_blender.json",
        "/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/cvpr_blender_pinerf_results"
        "/pose_eval_inerf_pinerf_blender.json",
    )
