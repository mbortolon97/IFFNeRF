from dataLoader import dataset_dict
from opt import config_parser
from renderer import *
from utils import *
from utils import flops_to_string, format_time

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
        return self.ids[self.curr:self.curr + self.batch]


def forward_step(tensorf, rays, color_bkgd_aug, batch_size=2048, n_samples=1024, ndc_ray=False, device='cpu',
                 is_train=True):
    if color_bkgd_aug == "random":
        bg_color = torch.rand(3, device=tensorf.device)
    elif color_bkgd_aug == "white":
        bg_color = torch.ones(3, device=tensorf.device)
    elif color_bkgd_aug == "black":
        bg_color = torch.zeros(3, device=tensorf.device)
    # if white_bg:
    #     bg_color = torch.ones(3, device=rgb_train.device)
    # else:
    #     bg_color = torch.zeros(3, device=rgb_train.device)

    # rgb_map, alphas_map, depth_map, weights, uncertainty
    renderer_results = renderer(rays, tensorf, chunk=batch_size, N_samples=n_samples, bg_color=bg_color,
                                ndc_ray=ndc_ray, device=device, is_train=True)

    return renderer_results, bg_color


def profile_network(args, warmup_steps=5):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsample_list = args.upsamp_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    n_samples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    tensorf = eval(args.model_name)(aabb, reso_cur, device,
                                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                                    app_dim=args.data_dim_color, near_far=near_far,
                                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                                    density_shift=args.density_shift, distance_scale=args.distance_scale,
                                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                                    featureC=args.featureC, step_ratio=args.step_ratio,
                                    fea2denseAct=args.fea2denseAct, contraction_type=args.contraction_type)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logarithmic space
    n_voxel_list = (torch.round(torch.exp(
        torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final),
                       len(upsample_list) + 1))).long()).tolist()[
                   1:]

    torch.cuda.empty_cache()
    psnrs, psnrs_test = [], [0]

    if hasattr(train_dataset, 'color_bkgd_aug'):
        color_bkgd_aug = train_dataset.color_bkgd_aug
    elif white_bg:
        color_bkgd_aug = 'white'
    else:
        color_bkgd_aug = 'black'

    allrays = train_dataset.all_rays
    if hasattr(train_dataset, 'all_rgbs'):
        allrgbs = train_dataset.all_rgbs
    else:
        allrgbs = train_dataset.all_rgba

    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    training_sampler = SimpleSampler(allrays.shape[0], args.batch_size)

    ortho_reg_weight = args.Ortho_weight
    print("initial ortho_reg_weight", ortho_reg_weight)

    l1_reg_weight = args.L1_weight_inital
    print("initial l1_reg_weight", l1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    ray_idx = training_sampler.nextids()
    rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

    # warm up cuda memory allocator, recommended here: https://github.com/pytorch/pytorch/blob/master/torch/autograd/profiler.py
    for i in range(warmup_steps):
        outputs = forward_step(tensorf, rays_train, color_bkgd_aug, batch_size=args.batch_size, n_samples=n_samples,
                               ndc_ray=ndc_ray, device=device, is_train=True)

    ### forward only
    with torch.no_grad(), torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True) as prof_forward:
        outputs = forward_step(tensorf, rays_train, color_bkgd_aug, batch_size=args.batch_size, n_samples=n_samples,
                               ndc_ray=ndc_ray, device=device, is_train=True)

    events_forward = prof_forward.events()
    forward_flops = sum([int(evt.flops) for evt in events_forward])

    ### forward timing step
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        outputs = forward_step(tensorf, rays_train, color_bkgd_aug, batch_size=args.batch_size, n_samples=n_samples,
                               ndc_ray=ndc_ray, device=device, is_train=False)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        forward_time = start.elapsed_time(end)

    ### forward + backward
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True) as prof_all:

        training_step(TV_weight_app, TV_weight_density, args, color_bkgd_aug, l1_reg_weight, lr_factor, n_samples,
                      ndc_ray, optimizer, ortho_reg_weight, rays_train, rgb_train, tensorf, tvreg)

    events_all = prof_all.events()
    all_flops = sum([int(evt.flops) for evt in events_all])

    ### training timing step
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    training_step(TV_weight_app, TV_weight_density, args, color_bkgd_aug, l1_reg_weight, lr_factor, n_samples,
                  ndc_ray, optimizer, ortho_reg_weight, rays_train, rgb_train, tensorf, tvreg)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    all_time = start.elapsed_time(end)

    print("forward flops: ", flops_to_string(forward_flops))
    print("only backward flops (estimated):", flops_to_string(all_flops - forward_flops))
    print("total iteration flops: ", flops_to_string(all_flops))

    print("forward time: ", format_time(forward_time))
    print("only backward time (estimated):", format_time(all_time - forward_time))
    print("total iteration time: ", format_time(all_time))


def training_step(TV_weight_app, TV_weight_density, args, color_bkgd_aug, l1_reg_weight, lr_factor, n_samples, ndc_ray,
                  optimizer, ortho_reg_weight, rays_train, rgb_train, tensorf, tvreg):
    (rgb_map, alphas_map, depth_map, weights, uncertainty), bg_color = forward_step(tensorf, rays_train,
                                                                                    color_bkgd_aug,
                                                                                    batch_size=args.batch_size,
                                                                                    n_samples=n_samples,
                                                                                    ndc_ray=ndc_ray, device=device,
                                                                                    is_train=True)
    with torch.no_grad():
        if rgb_train.shape[-1] > 3:
            rgb_train = rgb_train[..., :3] * rgb_train[..., -1:] + bg_color * (1.0 - rgb_train[..., -1:])
            rgb_train = rgb_train.clamp(0, 1)
    loss = torch.mean((rgb_map - rgb_train) ** 2)
    # loss
    total_loss = loss
    if ortho_reg_weight > 0:
        loss_reg = tensorf.vector_comp_diffs()
        total_loss += ortho_reg_weight * loss_reg
    if l1_reg_weight > 0:
        loss_reg_L1 = tensorf.density_L1()
        total_loss += l1_reg_weight * loss_reg_L1
    if TV_weight_density > 0:
        TV_weight_density *= lr_factor
        loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
        total_loss = total_loss + loss_tv
    if TV_weight_app > 0:
        TV_weight_app *= lr_factor
        loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
        total_loss = total_loss + loss_tv
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    profile_network(args)
