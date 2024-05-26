import os

def get_highest_valid_checkpoint(root_dir):
    sorted_filenames = sorted(os.listdir(root_dir), reverse=True)
    for file_name in sorted_filenames:
        ckpt_filepath = os.path.join(root_dir, file_name)
        if os.path.isfile(ckpt_filepath) and ckpt_filepath.endswith(".th"):
            return ckpt_filepath
    return ""

def parse_exp_dir(exp_dir, suffix):
    objects_checkpoints = {}
    exp_dirs_filenames = os.listdir(exp_dir)
    for exp_dir_filename in exp_dirs_filenames:
        exp_dir_filepath = os.path.join(exp_dir, exp_dir_filename)
        if not (
            os.path.isdir(exp_dir_filepath)
            and exp_dir_filename.startswith("tensorf_")
            and exp_dir_filename.endswith(suffix)
        ):
            continue
        name_components = exp_dir_filepath.split("_")
        sequence_id = name_components[-2]
        category_name = ""
        checkpoint_filepath = get_highest_valid_checkpoint(exp_dir_filepath)
        if checkpoint_filepath == "":
            print(
                f"Object {sequence_id} of category {category_name} skipped because no valid checkpoint found"
            )
            continue
        objects_checkpoints[sequence_id] = {
            "checkpoint_filepath": checkpoint_filepath,
            "sequence_id": sequence_id,
            "category_name": category_name,
        }
    return objects_checkpoints