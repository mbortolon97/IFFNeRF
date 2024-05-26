from typing import Optional, Callable, Tuple
import torch
from pose_estimation.test import test_pose_estimation
from pose_estimation.loss import DistanceBasedScoreLoss
from torch.utils.tensorboard import SummaryWriter
from pose_estimation.identification_module import IdentificationModule

def train_id_module(
    ckpt_path,
    device,
    id_module: IdentificationModule,
    rays_generator: Optional[
        Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ],
    train_dataset,
    val_dataset,
    sequence_id,
    start_iterations: int = 0,
    renewal_every_n_iterations: int = 10,
    inerf_refinement=False,
    nerf_model=None,
    display_every_n_iterations: int = 20,
    val_every_n_iterations: int = 20,
    n_iterations: int = 1500,
    gradient_accumulation_steps: int = 32,
    lock_backbone: bool = True,
):
    id_module.train()
    if lock_backbone:
        id_module.image_preprocessing_net.eval()
    optimizer = torch.optim.Adam(
        [
            {
                "params": id_module.ray_preprocessor.parameters(),
                "lr": 4.0e-3,
            },
            {
                "params": id_module.attention.parameters(),
                "lr": 4.0e-3,
            },
            {
                "params": id_module.image_preprocessing_net.parameters(),
                "lr": 1.0e-3,
            },
        ],
        lr=1.0e-3,
    )
    loss_fn = DistanceBasedScoreLoss()
    running_loss = 0.0

    # Initialize the SummaryWriter for TensorBoard
    # Its output will be written to ./runs/
    writer = SummaryWriter()
    writer.add_text("config/ckpt_path", ckpt_path)
    writer.add_text("config/train_dataset_root_dir", train_dataset.root_dir)

    best_model = None
    best_angular_score = 360.0

    model_up = torch.mean(train_dataset.poses[:, :3, 1], dim=0).to(device=device)

    rays_ori, rays_dirs, rays_rgb = None, None, None
    target_scores = None
    for iteration in range(start_iterations, n_iterations):
        if iteration % renewal_every_n_iterations == 0:
            rays_ori, rays_dirs, rays_rgb = rays_generator()
        optimizer.zero_grad()

        # random data extraction
        img_idx = torch.randint(
            0,
            train_dataset.all_rgbs.shape[0],
            (gradient_accumulation_steps,),
            dtype=torch.long,
            device=train_dataset.all_rgbs.device,
        )

        accumulation_loss = 0.0
        for gradient_accumulation_step in range(gradient_accumulation_steps):
            train_img = train_dataset.all_rgbs[img_idx[gradient_accumulation_step]].to(
                device, non_blocking=True
            )
            if train_img.shape[-1] == 4:
                train_img_mask = train_img[..., -1] > 0.3
                train_img = torch.multiply(train_img[..., :3], train_img[..., -1:]) + (
                    1 - train_img[..., -1:]
                )
            else:
                train_img_mask = torch.ones_like(train_img[..., -1], dtype=torch.bool)

            target_camera_pose = train_dataset.poses[
                img_idx[gradient_accumulation_step]
            ].to(device, non_blocking=True)
            target_camera_intrinsic = train_dataset.K.to(device, non_blocking=True)[0]

            # Make predictions for this batch
            scores, attn_map, img_features, rays_idx = id_module(
                train_img, train_img_mask, rays_ori, -rays_dirs, rays_rgb
            )

            loss_score, target_scores = loss_fn(
                scores,
                # features_img_w_pe_flat,
                # features_rays,
                target_camera_pose,
                target_camera_intrinsic,
                rays_ori[rays_idx],
                -rays_dirs[rays_idx],
                attn_map.shape[-2],
                id_module.backbone_wh,
                model_up=model_up,
            )

            if loss_score.isnan().any():
                continue

            # Compute the loss and its gradients
            loss = loss_score / gradient_accumulation_steps
            loss.backward()

            # print(loss)

            accumulation_loss += loss.item()

        # Adjust learning weights
        optimizer.step()

        writer.add_scalar("train/loss", accumulation_loss, global_step=iteration)

        # Gather data and report
        running_loss += accumulation_loss
        if iteration % display_every_n_iterations == display_every_n_iterations - 1:
            last_loss = running_loss / display_every_n_iterations  # loss per batch
            print(f"[{iteration}] loss: {last_loss}")
            running_loss = 0.0

        if iteration % val_every_n_iterations == val_every_n_iterations - 1:
            print("Eval on train...")
            (
                _,
                train_avg_translation_error,
                train_avg_angular_error,
                train_avg_score,
                train_recall,
            ) = test_pose_estimation(
                train_dataset,
                id_module,
                rays_ori,
                rays_dirs,
                rays_rgb,
                model_up,
                sequence_id=sequence_id,
                loss_fn=loss_fn,
                inerf_refinement=inerf_refinement,
                nerf_model=nerf_model,
                # save=True,
                # save=(195 < iteration < 210),
            )

            writer.add_scalar(
                "train/avg_translation_error",
                train_avg_translation_error,
                global_step=iteration,
            )
            writer.add_scalar(
                "train/avg_angular_error",
                train_avg_angular_error,
                global_step=iteration,
            )
            writer.add_scalar(
                "train/avg_loss_score",
                train_avg_score,
                global_step=iteration,
            )
            writer.add_scalar(
                "train/recall",
                train_recall,
                global_step=iteration,
            )

            print("Eval on validation...")
            (
                _,
                val_avg_translation_error,
                val_avg_angular_error,
                val_avg_score,
                val_recall,
            ) = test_pose_estimation(
                val_dataset,
                id_module,
                rays_ori,
                rays_dirs,
                rays_rgb,
                model_up,
                sequence_id=sequence_id,
                loss_fn=loss_fn,
                inerf_refinement=inerf_refinement,
                nerf_model=nerf_model,
            )

            writer.add_scalar(
                "val/avg_translation_error",
                val_avg_translation_error,
                global_step=iteration,
            )
            writer.add_scalar(
                "val/avg_angular_error",
                val_avg_angular_error,
                global_step=iteration,
            )
            writer.add_scalar(
                "val/avg_loss_score",
                val_avg_score,
                global_step=iteration,
            )
            writer.add_scalar(
                "val/recall",
                val_recall,
                global_step=iteration,
            )

            id_module.train()
            if lock_backbone:
                id_module.image_preprocessing_net.eval()

    torch.save(
        {
            "epoch": n_iterations,
            "model_state_dict": id_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "running_loss": running_loss,
        },
        ckpt_path,
    )
