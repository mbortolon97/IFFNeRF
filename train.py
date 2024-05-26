import datetime
from time import time
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

from dataLoader import dataset_dict
from opt import config_parser
from renderer import *
from utils import *
from models.tensoRF import TensorVM, TensorVMSplit, TensorCP
from utils import format_time
from torch_efficient_distloss import eff_distloss

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
        alpha.cpu(), f"{args.ckpt[:-3]}.ply", bbox=tensorf.aabb.cpu(), level=0.005
    )


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!!")
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir, split="train", downsample=args.downsample_train, is_stack=True
        )
        PSNRs_test = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/{args.expname}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/{args.expname}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f"{logfolder}/{args.expname}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/{args.expname}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )


def reconstruction(args, return_result=False, report_function=None):
    if return_result:
        assert args.render_test, "with return results also test must be executed"
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        args.datadir, split="train", downsample=args.downsample_train, is_stack=False
    )
    test_dataset = dataset(
        args.datadir, split="test", downsample=args.downsample_train, is_stack=True
    )
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsample_list = args.upsamp_list
    update_alpha_mask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{args.expname}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    n_samples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            aabb,
            reso_cur,
            device,
            density_n_comp=n_lamb_sigma,
            appearance_n_comp=n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=near_far,
            shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=args.fea_pe,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct,
            contraction_type=args.contraction_type,
        )

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logarithmic space
    n_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsample_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]

    torch.cuda.empty_cache()
    psnrs, psnrs_test = [], [0]

    if hasattr(train_dataset, "color_bkgd_aug"):
        color_bkgd_aug = train_dataset.color_bkgd_aug
    elif white_bg:
        color_bkgd_aug = "white"
    else:
        color_bkgd_aug = "black"

    allrays = train_dataset.all_rays
    if hasattr(train_dataset, "all_rgbs"):
        allrgbs = train_dataset.all_rgbs
    else:
        allrgbs = train_dataset.all_rgba

    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)

    train_batch_size = args.train_batch_size
    if train_batch_size < 1:
        train_batch_size = args.batch_size
    training_sampler = SimpleSampler(allrays.shape[0], train_batch_size)

    ortho_reg_weight = args.Ortho_weight
    print("initial ortho_reg_weight", ortho_reg_weight)

    l1_reg_weight = args.L1_weight_inital
    print("initial l1_reg_weight", l1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    distortion_loss = eff_distloss

    #  training timing step
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    start_time = time()
    pbar = tqdm(
        range(args.n_iters),
        miniters=args.progress_refresh_rate,
        file=sys.stdout,
        disable=False,
    )
    for iteration in pbar:
        ray_idx = training_sampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        if color_bkgd_aug == "random":
            bg_color = torch.rand(3, device=rgb_train.device)
        elif color_bkgd_aug == "white":
            bg_color = torch.ones(3, device=rgb_train.device)
        elif color_bkgd_aug == "black":
            bg_color = torch.zeros(3, device=rgb_train.device)
        # if white_bg:
        #     bg_color = torch.ones(3, device=rgb_train.device)
        # else:
        #     bg_color = torch.zeros(3, device=rgb_train.device)

        if rgb_train.shape[-1] > 3:
            rgb_train = rgb_train[..., :3] * rgb_train[..., -1:] + bg_color * (
                1.0 - rgb_train[..., -1:]
            )
            rgb_train = rgb_train.clamp(0, 1)

        # rgb_map, alphas_map, depth_map, weights, uncertainty

        rgb_map, depth_map, acc_map, weights, z_vals, dists = tensorf(
            rays_train.to(device),
            N_samples=n_samples,
            bg_color=bg_color,
            ndc_ray=ndc_ray,
            is_train=True,
        )

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # sparsity loss

        # loss
        total_loss = loss
        if ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += ortho_reg_weight * loss_reg
            summary_writer.add_scalar(
                "train/reg", loss_reg.detach().item(), global_step=iteration
            )
        if l1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += l1_reg_weight * loss_reg_L1
            summary_writer.add_scalar(
                "train/reg_l1", loss_reg_L1.detach().item(), global_step=iteration
            )

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_density", loss_tv.detach().item(), global_step=iteration
            )
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
            )

        # dist_loss_comp = distortion_loss(weights, z_vals, dists)
        dist_loss_comp = torch.exp(weights.abs()).mean()
        total_loss = total_loss + dist_loss_comp * 0.1
        summary_writer.add_scalar(
            "train/distortion_loss",
            dist_loss_comp.detach().item(),
            global_step=iteration,
        )
        # total_loss = total_loss +

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        psnrs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar("train/PSNR", psnrs[-1], global_step=iteration)
        summary_writer.add_scalar("train/mse", loss, global_step=iteration)
        # summary_writer.add_scalar("train/dist", dist_loss_comp, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"Iteration {iteration:05d}:"
                + f" train_psnr = {float(np.mean(psnrs)):.2f}"
                + f" test_psnr = {float(np.mean(psnrs_test)):.2f}"
                + f" mse = {loss:.6f}"
            )
            psnrs = []

        if iteration % 30 and report_function is not None:
            end_time = time()
            intermediate_train_time = (end_time - start_time) * 1000
            report_function(intermediate_train_time, iteration)

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            psnrs_test = evaluation(
                test_dataset,
                tensorf,
                args,
                renderer,
                f"{logfolder}/imgs_vis/",
                N_vis=args.N_vis,
                prtx=f"{iteration:06d}_",
                N_samples=n_samples,
                white_bg=white_bg,
                ndc_ray=ndc_ray,
                compute_extra_metrics=False,
            )
            summary_writer.add_scalar(
                "test/psnr", np.mean(psnrs_test), global_step=iteration
            )

        if iteration in update_alpha_mask_list:
            if (
                reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
            ):  # update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_alpha_mask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                l1_reg_weight = args.L1_weight_rest
                print("continuing l1_reg_weight", l1_reg_weight)

            if not args.ndc_ray and iteration == update_alpha_mask_list[1]:
                # filter rays outside the bbox
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                training_sampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if iteration in upsample_list:
            n_voxels = n_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            n_samples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(
                args.lr_init * lr_scale, args.lr_basis * lr_scale
            )
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    total_training_time = start.elapsed_time(end)
    time_per_iteration = total_training_time / (
        args.n_iters + 1 if args.n_iters == 0 else 0
    )
    print("total training time: ", format_time(total_training_time))
    print("time per iteration: ", format_time(time_per_iteration))

    tensorf.save(f"{logfolder}/{args.expname}.th")

    test_metrics = {}
    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir, split="train", downsample=args.downsample_train, is_stack=True
        )
        save_path = f"{logfolder}/imgs_train_all/"
        psnrs_test = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            save_path,
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(psnrs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        save_path = f"{logfolder}/imgs_test_all/"
        save_path = None
        test_results = evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            save_path,
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            return_result=return_result,
        )
        if return_result:
            psnrs_test, test_metrics = test_results
        else:
            psnrs_test = test_results
        summary_writer.add_scalar(
            "test/psnr_all", np.mean(psnrs_test), global_step=iteration
        )
        print(
            f"======> {args.expname} test all psnr: {np.mean(psnrs_test)} <========================"
        )

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print("========>", c2ws.shape)
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )

    if return_result:
        return {
            "total_train_time": total_training_time,
            "total_test_time": test_metrics["total_test_time"],
            "test_psnr": test_metrics["avg_psnr"],
        }


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
