import torch
from pose_estimation import sampling

def load_model(checkpoint_path, device):
    # Load TensoRF
    ckpt = torch.load(checkpoint_path, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(ckpt["model_name"])(**kwargs)
    tensorf.load(ckpt)

    for param in tensorf.parameters():
        param.requires_grad = False
    return tensorf


def cache_model_on_gpu(ckpt_path, device):
    model = load_model(ckpt_path, device)
    model.eval()
    model = model.to(device, non_blocking=True)

def explore_model(model, gen_points: int = 20000):
    samples = sampling.iterative_surface_sampling_process(
        model, gen_points=gen_points, n_iteration=4, max_resampling_iterations=200
    )
    sample_normals = sampling.samples_points_normals(model, samples)
    point, dirs, rgb = sampling.generate_all_possible_rays(
        samples,
        sample_normals,
        model,
    )

    return point, dirs, rgb