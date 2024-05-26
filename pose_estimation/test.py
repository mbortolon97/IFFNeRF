import torch
import numpy as np
from pose_estimation.pose_geometry import make_rotation_mat, compute_line_intersection_impl2, exclude_negatives
import time
from tqdm import tqdm
from statistics import mean
from inerf.estimate_pose_inerf import pose_estimation
from pose_estimation.errors import compute_angular_error, compute_translation_error

def test_pose_estimation(
    dataset,
    id_module,
    rays_ori,
    rays_dirs,
    rays_rgb,
    model_up,
    sequence_id="",
    loss_fn=None,
    save=False,
    inerf_refinement=False,
    nerf_model=None,
    save_all=False,
    augmentation_parameters={},
):
    id_module.eval()

    translation_errors = []
    angular_errors = []
    model_up = torch.divide(model_up, torch.linalg.norm(model_up, dim=-1, keepdim=True))

    test_cams = [i for i in range(len(dataset.all_rgbs))]

    destination_cam_idx = torch.tensor(
        [i for _ in range(1) for i in test_cams],
        dtype=torch.long,
        device=dataset.all_rgbs.device,
    )

    # augment the images
    obs_imgs = []
    new_cam_Ks = []
    for cam_idx in destination_cam_idx:
        test_img = dataset.all_rgbs[cam_idx]
        obs_img = test_img.cpu().numpy()
        new_cam_K = dataset.K.cpu().numpy()

        obs_imgs.append(obs_img)
        new_cam_Ks.append(new_cam_K)
    np_test_cam_Ks = np.stack(new_cam_Ks)
    # assert np_test_cam_Ks.shape == dataset.K.shape, 'shape must be the same'
    new_cam_K = torch.from_numpy(np_test_cam_Ks).to(
        dtype=dataset.K.dtype, device=dataset.K.device
    )
    np_test_imgs = np.stack(obs_imgs)
    # assert np_test_imgs.shape == dataset.all_rgbs.shape, 'shape must be the same'
    dataset.all_rgbs = torch.from_numpy(np_test_imgs).to(
        dtype=dataset.all_rgbs.dtype, device=dataset.all_rgbs.device
    )

    dataset.all_rays = dataset.all_rays[destination_cam_idx]
    dataset.poses = dataset.poses[destination_cam_idx]

    recalls = []
    avg_loss_scores = []
    results = []
    start_time = time.time()
    for img_idx in tqdm(range(dataset.all_rgbs.shape[0])):
        pose = dataset.poses[img_idx].to(rays_ori.device, non_blocking=True)
        target_camera_intrinsic = dataset.K.to(rays_ori.device, non_blocking=True)[0]

        if augmentation_parameters is not None:
            pass

        obs_img = dataset.all_rgbs[img_idx].to(rays_ori.device, non_blocking=True)

        if obs_img.shape[-1] == 4:
            mask_img = obs_img[..., -1]
            obs_img = torch.multiply(obs_img[..., :3], obs_img[..., -1:]) + (
                1 - obs_img[..., -1:]
            )
        else:
            mask_img = torch.ones_like(obs_img[..., -1], dtype=torch.bool)

        idx, weights, pred_scores, attention_map = id_module.test_image(
            obs_img,
            mask_img,
            rays_ori,
            rays_dirs,
            rays_rgb,
            rays_to_output=100,
        )

        if save and (img_idx == 0 or save_all):
            saving_dict = {
                "gt_pose": pose.cpu(),
                "camera_intrinsic": target_camera_intrinsic.cpu(),
                "all_rays_ori": rays_ori.cpu(),
                "all_rays_dirs": rays_dirs.cpu(),
                "all_rays_rgb": rays_rgb.cpu(),
                "obs_img": obs_img.cpu(),
                "mask_img": mask_img.cpu(),
                "topk_nonunique_ray_idx": idx.cpu(),
                "topk_nonunique_weights": weights.cpu(),
                "all_predict_weights": pred_scores.cpu(),
            }

        avg_score = -1.0
        recall = -1.0
        if loss_fn is not None:
            avg_score, target_scores = loss_fn(
                pred_scores,
                pose,
                target_camera_intrinsic,
                rays_ori,
                rays_dirs,
                attention_map.shape[-2],
                id_module.backbone_wh,
                model_up=model_up,
            )
            avg_score = avg_score.item()
            target_idx = torch.topk(weights, k=100).indices
            intersection = torch.count_nonzero(torch.isin(target_idx, idx))
            recall = intersection.item() / target_idx.shape[0]

            if save and (img_idx == 0 or save_all):
                saving_dict["all_target_weights"] = target_scores.cpu()
                saving_dict["loss"] = avg_score
                saving_dict["recall"] = recall

        avg_loss_scores.append(avg_score)
        recalls.append(recall)

        unique_elements, counts = torch.unique(rays_ori[idx], return_counts=True, dim=0)
        mask = torch.isin(
            rays_ori[idx], unique_elements[counts == 1], assume_unique=True
        ).any(dim=1)
        idx = idx[mask]
        weights = weights[mask]

        if save and (img_idx == 0 or save_all):
            saving_dict["topk_unique_ray_idx"] = idx.cpu()
            saving_dict["topk_unique_weights"] = weights.cpu()

        weights = torch.divide(weights, torch.sum(weights))
        camera_optical_center = compute_line_intersection_impl2(
            rays_ori[idx], rays_dirs[idx]  # , weights=weights
        )
        weights = torch.multiply(
            weights,
            exclude_negatives(camera_optical_center, rays_ori[idx], rays_dirs[idx]),
        )
        weights = torch.divide(weights, torch.sum(weights))
        camera_optical_center = compute_line_intersection_impl2(
            rays_ori[idx], rays_dirs[idx]  # , weights=weights
        )
        # camera_optical_center = compute_line_intersection_impl2(
        #     rays_ori[idx], rays_dirs[idx]
        # )

        if torch.isnan(camera_optical_center).any(dim=0):
            print("camera_optical_center is nan")

        camera_watch_dir = torch.multiply(rays_dirs[idx], weights[:, None]).sum(dim=0)
        camera_watch_dir = torch.divide(
            camera_watch_dir, torch.linalg.norm(camera_watch_dir, dim=-1, keepdim=True)
        )

        c2w_matrix = torch.eye(4, dtype=rays_ori.dtype, device=rays_ori.device)
        w2c_rotation_matrix = make_rotation_mat(-camera_watch_dir, model_up)
        if torch.linalg.det(w2c_rotation_matrix) < 1.0e-7:
            print("extracted rotation matrix is singular")
            w2c_rotation_matrix = torch.eye(3)
        c2w_matrix[:3, :3] = torch.linalg.inv(w2c_rotation_matrix)
        c2w_matrix[:3, -1] = camera_optical_center
        # c2w_matrix[:3, :3] = torch.linalg.inv(R[0])
        # c2w_matrix[:3, -1] = t[0]

        if save and (img_idx == 0 or save_all):
            saving_dict["topk_unique_weights_after_exclusion"] = weights.cpu()
            saving_dict["pred_camera_optical_center"] = camera_optical_center.cpu()
            saving_dict["pred_camera_watch_dir"] = -camera_watch_dir.cpu()
            saving_dict["pred_c2w_matrix"] = c2w_matrix.cpu()
            saving_dict["model_up"] = model_up.cpu()

            torch.save(
                saving_dict,
                f"/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/sample_results_{img_idx}.th",
            )

            print("Sample result saved")

        if torch.isnan(c2w_matrix).any():
            print("wrong c2w")
            c2w_matrix = torch.eye(4, dtype=rays_ori.dtype, device=rays_ori.device)

        if inerf_refinement:
            _, c2w_matrix, poses_estimated = pose_estimation(
                c2w_matrix,
                torch.cat((obs_img, mask_img[..., None].to(obs_img.dtype)), dim=-1)
                .cpu()
                .numpy(),
                target_camera_intrinsic,
                nerf_model,
                device=c2w_matrix.device,
                n_iters=800,
                print_progress=False,
                lrate=0.02,
                dice_loss=True,
                sampling_strategy="random",
            )
            c2w_matrix = c2w_matrix.to(pose.device, non_blocking=True)

        gt_camera_position = (
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=pose.dtype, device=pose.device
            ).reshape(1, 4)
            @ pose[:3, :].T
        )

        pred_camera_position = (
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=pose.dtype, device=pose.device
            ).reshape(1, 4)
            @ c2w_matrix[:3, :].T
        )

        translation_error = compute_translation_error(
            gt_camera_position, pred_camera_position
        )
        angular_error = compute_angular_error(pose[:3, :3], c2w_matrix[:3, :3])

        translation_errors.append(translation_error.item())
        angular_errors.append(angular_error.item())

        results.append(
            {
                "sequence_id": sequence_id,
                "category_name": "id_net",
                "frame_id": img_idx,
                "loss": weights.mean().item(),
                "scores_loss": avg_score,
                "recall": recall,
                "total_optimization_time_in_ms": 0.0,
                "pred_c2w": c2w_matrix.cpu().tolist(),
                "gt_c2w": pose.cpu().tolist(),
            }
        )

    total_time = time.time() - start_time
    time_per_element = total_time / dataset.all_rgbs.shape[0]

    avg_loss_score = mean(avg_loss_scores)
    avg_recall = mean(recalls)
    print("Average loss score: ", avg_loss_score)
    print("Average Recall: ", avg_recall)
    print("Time per element: ", time_per_element)

    avg_translation_error = mean(translation_errors)
    avg_angular_error = mean(angular_errors)
    print("Translation Error: ", avg_translation_error)
    print("Angular Error: ", avg_angular_error)

    print(
        "Smallest translation error: ",
        min(range(len(translation_errors)), key=translation_errors.__getitem__),
    )

    return results, avg_translation_error, avg_angular_error, avg_loss_score, avg_recall
