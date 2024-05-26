import math
import time

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from models.ref import Ref
from utils import power_transformation

from .sh import eval_sh_bases


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1.0 - torch.exp(-sigma * dist)

    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def SHRender(xyz_sampled, viewdirs, features, *args):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb, None


def RGBRender(xyz_sampled, viewdirs, features, *args):
    rgb = features
    return rgb, None


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume, contraction_type="aabb"):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.contraction_type = contraction_type

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.tensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]],
            dtype=torch.int64,
        ).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(
            self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        if self.contraction_type == "unisphere":
            # From Zip-NeRF
            # [-inf, inf]^3 -> sphere of [0, 1]^3;
            # roi -> sphere of [0.25, 0.75]^3
            aabb_center = (self.aabb[0] + self.aabb[1]) / 2.0
            centered_xyz = xyz_sampled - aabb_center
            return power_transformation(centered_xyz, alpha=-1.5)
        else:
            return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1

    def march_alpha_grid(self, ray_origin: torch.Tensor, ray_direction: torch.Tensor):
        norm_ray_origin = self.normalize_coord(ray_origin)
        # Initialize the ray traversal variables
        current_cell = torch.floor(norm_ray_origin).long()
        step = torch.sign(ray_direction).long()
        t_max = torch.tensor(float("inf"))

        # Compute the maximum t-value for each axis
        for i in range(3):
            if ray_direction[i] != 0:
                if step[i] > 0:
                    t_max[i] = (
                        current_cell[i] + 1 - norm_ray_origin[i]
                    ) / ray_direction[i]
                else:
                    t_max[i] = (current_cell[i] - norm_ray_origin[i]) / ray_direction[i]

        # Initialize the counter for the number of cells traversed
        num_cells_traversed = 0

        # Traverse the grid using the DDA algorithm
        while True:
            # Find the axis with the smallest t-value
            min_axis = torch.argmin(t_max)

            # Increment the corresponding current cell and update t_max
            current_cell[min_axis] += step[min_axis]
            t_max[min_axis] += torch.abs(1.0 / ray_direction[min_axis])

            # Check if we are still within the grid boundaries
            if (
                current_cell[min_axis] < 0
                or current_cell[min_axis] >= grid_size[min_axis]
            ):
                break

            # Increment the number of cells traversed
            num_cells_traversed += 1

        return num_cells_traversed
        pass


class Gaussian(torch.nn.Module):
    def __init__(self, sigma: float = 0.1):
        super().__init__()

        self.denom = 2 * sigma**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2) / self.denom)


class MLPRender_Gaussian(torch.nn.Module):
    def __init__(self, inChanel):
        super(MLPRender_Gaussian, self).__init__()

        self.in_mlpC = 3 + inChanel
        layer1 = torch.nn.Linear(self.in_mlpC, self.in_mlpC)
        layer2 = torch.nn.Linear(self.in_mlpC, self.in_mlpC)
        layer3 = torch.nn.Linear(self.in_mlpC, self.in_mlpC)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.CELU(inplace=True),
            layer2,
            torch.nn.CELU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, *args):
        indata = [features, viewdirs]
        mlp_in = torch.cat(indata, dim=-1)
        raw = self.mlp(mlp_in)
        rgb = torch.sigmoid(raw[..., :3])

        return rgb, raw[..., 3:]


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, *args):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb, None


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, *args):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb, None


class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, *args):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb, None


class TensorBase(torch.nn.Module):
    def __init__(
        self,
        aabb,
        gridSize,
        device,
        density_n_comp=8,
        appearance_n_comp=24,
        app_dim=27,
        shadingMode="MLP_PE",
        alphaMask=None,
        near_far=[2.0, 6.0],
        density_shift=-10,
        alphaMask_thres=0.001,
        distance_scale=25,
        rayMarch_weight_thres=0.0001,
        pos_pe=6,
        view_pe=6,
        fea_pe=6,
        featureC=128,
        step_ratio=2.0,
        fea2denseAct="softplus",
        contraction_type="aabb",
        step_size_bg=0.1,
    ):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.contraction_type = contraction_type

        self.step_size_bg = step_size_bg

        self.update_stepSize(gridSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]

        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = (
            shadingMode,
            pos_pe,
            view_pe,
            fea_pe,
            featureC,
        )
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

        self.it = 0

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == "MLP_PE":
            self.renderModule = MLPRender_PE(
                self.app_dim, view_pe, pos_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea":
            self.renderModule = MLPRender_Fea(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP":
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == "Ref":
            self.renderModule = Ref(
                self.app_dim, viewpe=view_pe, feature_c=featureC
            ).to(device)
        elif shadingMode == "SH":
            self.renderModule = SHRender
        elif shadingMode == "RGB":
            assert self.app_dim == 3
            self.renderModule = RGBRender
        elif shadingMode == "MLP_GARF":
            self.renderModule = MLPRender_Gaussian(self.app_dim).to(device)
        else:
            print("Unrecognized shading module")
            exit()

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.tensor(gridSize, dtype=torch.long, device=self.device)
        aabb_grid_size = self.gridSize
        if self.contraction_type == "unisphere":
            aabb_grid_size = (
                aabb_grid_size * 0.5
            )  # in the first half it is equivalent to a linear function
        self.units = self.aabbSize / (aabb_grid_size - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        near, far = self.near_far
        if self.contraction_type == "unisphere":
            self.n_samples_bg = (far - near) / self.step_size_bg
        else:
            self.n_samples_bg = 0
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        if self.contraction_type == "unisphere":
            # From Zip-NeRF
            # [-inf, inf]^3 -> sphere of [0, 1]^3;
            # roi -> sphere of [0.25, 0.75]^3
            aabb_center = (self.aabb[0] + self.aabb[1]) / 2.0
            centered_xyz = xyz_sampled - aabb_center
            return power_transformation(centered_xyz, alpha=-1.5)
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            "aabb": self.aabb,
            "gridSize": self.gridSize.tolist(),
            "density_n_comp": self.density_n_comp,
            "appearance_n_comp": self.app_n_comp,
            "app_dim": self.app_dim,
            "contraction_type": self.contraction_type,
            "density_shift": self.density_shift,
            "alphaMask_thres": self.alphaMask_thres,
            "distance_scale": self.distance_scale,
            "rayMarch_weight_thres": self.rayMarch_weight_thres,
            "fea2denseAct": self.fea2denseAct,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "shadingMode": self.shadingMode,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "featureC": self.featureC,
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        model_name = type(self).__name__
        assert model_name in [
            "TensorVMSplit",
            "TensorCP",
        ], f"model name found {model_name}, expected 'TensorCP' or 'TensorVMSplit'"
        # model_name =
        ckpt = {
            "model_name": model_name,
            "kwargs": kwargs,
            "state_dict": self.state_dict(),
        }
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({"alphaMask.shape": alpha_volume.shape})
            ckpt.update({"alphaMask.mask": np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({"alphaMask.aabb": self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if "alphaMask.aabb" in ckpt.keys():
            length = np.prod(ckpt["alphaMask.shape"])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt["alphaMask.mask"])[:length].reshape(
                    ckpt["alphaMask.shape"]
                )
            )
            self.alphaMask = AlphaGridMask(
                self.device,
                ckpt["alphaMask.aabb"].to(self.device),
                alpha_volume.float().to(self.device),
                contraction_type=self.contraction_type,
            )
        self.load_state_dict(ckpt["state_dict"])

    def sample_ray_ndc(self, rays_o, rays_d, radii, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_infinity(self, rays_o, rays_d, radii, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        near_in_ndc = 1.0 / near
        far_in_ndc = 1.0 / far
        interpx = torch.linspace(near_in_ndc, 1e-7, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx = torch.clamp(
                interpx + (torch.rand_like(interpx).to(rays_o) / N_samples), 1e-8, 1.0
            )

        ndc_to_real_world = 1.0 / (1.0 - interpx)

        rays_pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * ndc_to_real_world[..., None]
        )
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, radii, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        N_samples_bg = self.n_samples_bg
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(
            N_samples + self.n_samples_bg, dtype=rays_o.dtype, device=rays_o.device
        )
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])

        if self.contraction_type == "unisphere":
            # TODO: add points
            individual_stepsizes = torch.repeat_interleave(
                torch.tensor(
                    [stepsize, self.step_size_bg],
                    dtype=rays_o.dtype,
                    device=rays_o.device,
                ),
                torch.tensor(
                    [N_samples + 1, self.n_samples_bg],
                    dtype=torch.long,
                    device=rays_o.device,
                ),
            )
            step = torch.multiply(individual_stepsizes[None], rng)
        else:
            step = torch.multiply(stepsize, rng)

        interpx = t_min[..., None] + step

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_zip_nerf(self, rays_o, rays_d, radii, is_train=True, N_samples=-1):
        six_hexagonal_pattern = torch.tensor(
            [
                0,
                (2 * math.pi) / 3,
                (4 * math.pi) / 3,
                (3 * math.pi) / 3,
                (5 * math.pi) / 3,
                math.pi / 3,
            ],
            dtype=rays_o.dtype,
            device=rays_o.device,
        )

        # divide by six because additional point will be added by the hexagonal pattern
        N_samples = N_samples // 6 if N_samples > 0 else self.nSamples // 6
        N_samples_bg = self.n_samples_bg
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(
            N_samples + self.n_samples_bg, dtype=rays_o.dtype, device=rays_o.device
        )
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])

        if self.contraction_type == "unisphere":
            # TODO: add points
            individual_stepsizes = torch.repeat_interleave(
                torch.tensor(
                    [stepsize, self.step_size_bg],
                    dtype=rays_o.dtype,
                    device=rays_o.device,
                ),
                torch.tensor(
                    [N_samples + 1, self.n_samples_bg],
                    dtype=torch.long,
                    device=rays_o.device,
                ),
            )
            step = torch.multiply(individual_stepsizes[None], rng)
        else:
            step = torch.multiply(stepsize, rng)

        interpx = t_min[..., None] + step

        t_dist = (interpx[:, 1:] - interpx[:, :-1]) / 2.0
        squared_t_dist = torch.square(t_dist)
        t_mean = (interpx[:, 1:] + interpx[:, :-1]) / 2.0
        squared_t_mean = torch.square(t_mean)

        breakpoint()

        j = torch.arange(
            0, six_hexagonal_pattern.shape[0], dtype=t_dist.dtype, device=t_dist.device
        )
        j = j[None, :].expand(t_dist.shape[0])
        interpx + (
            (
                t_dist
                * (
                    torch.square(interpx[:, 1:])
                    + 2 * torch.square(t_mean)
                    + (3.0 / math.sqrt(7.0))
                    * ((2 * j) / 5 - 1.0)
                    * torch.sqrt(
                        torch.square(squared_t_dist - squared_t_mean)
                        + 4 * torch.pow(t_mean, 4)
                    )
                )
            )
            / (squared_t_dist + 3 * squared_t_mean)
        )
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )

        return rays_pts, interpx, ~mask_outbbox

    def sample_point_color(self, rays_o, rays_d, radii, N_samples=20, **kwargs):
        before_n_samples = N_samples // 2
        after_n_samples = N_samples - before_n_samples
        step_size = self.stepSize

        rng = torch.arange(
            -before_n_samples, after_n_samples, dtype=rays_o.dtype, device=rays_o.device
        )[None]
        step = step_size * rng

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * step[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )

        return rays_pts, step, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize
        print(gridSize)

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize).view(
                (gridSize[1], gridSize[2])
            )
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(
            gridSize[::-1]
        )
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(
            self.device, self.aabb, alpha, contraction_type=self.contraction_type
        )

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(
            f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100)
        )
        return new_aabb

    @torch.no_grad()
    def filtering_rays(
        self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False
    ):
        print("========> filtering rays ...")
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if rays_chunk.shape[-1] > 6:
                radii = rays_chunk[..., -1]
            else:
                radii = None

            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_ray(
                    rays_o, rays_d, radii, N_samples=N_samples, is_train=False
                )
                mask_inbbox = (
                    self.alphaMask.sample_alpha(xyz_sampled).view(
                        xyz_sampled.shape[:-1]
                    )
                    > 0
                ).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(
            f"Ray filtering done! takes {time.time() - tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}"
        )
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    def forward(
        self,
        rays_chunk,
        white_bg=False,
        bg_color=None,
        is_train=False,
        ndc_ray=False,
        sample_func=None,
        N_samples=-1,
    ):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if rays_chunk.shape[-1] > 6:
            radii = rays_chunk[..., -1]
        else:
            radii = None

        if sample_func is not None:
            xyz_sampled, z_vals, ray_valid = sample_func(
                rays_chunk[:, :3],
                viewdirs,
                radii,
                is_train=is_train,
                N_samples=N_samples,
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
        elif ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3],
                viewdirs,
                radii,
                is_train=is_train,
                N_samples=N_samples,
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3],
                viewdirs,
                radii,
                is_train=is_train,
                N_samples=N_samples,
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        # breakpoint()
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres
        # app_mask = torch.full_like(weight, True, dtype=torch.bool)

        # if self.it % 200 == 0 and self.training:
        #     with torch.no_grad():
        #         from visual import display_point_cloud
        #
        #         ray_to_select = torch.randint(
        #             0,
        #             xyz_sampled.shape[0],
        #             size=(80,),
        #             dtype=torch.long,
        #             device=xyz_sampled.device,
        #         )
        #         display_point_cloud(xyz_sampled.view(-1, 3).cpu().numpy())
        #     self.it += 1
        # elif self.training:
        #     self.it += 1

        # viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        # cum_rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        cum_app_features = torch.zeros(
            (*xyz_sampled.shape[:2], self.app_dim), device=xyz_sampled.device
        )

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            cum_app_features[app_mask] = app_features

            # cum_rgb[app_mask], _ = self.renderModule(
            #     None,
            #     viewdirs[app_mask],
            #     app_features,
            #     None,
            # )
        rays_to_consider = app_mask.any(dim=-1)
        acc_map = torch.sum(weight, -1)
        cum_app_features = torch.sum(weight[..., None] * cum_app_features, -2)
        # rgb_map = torch.sum(weight[..., None] * cum_rgb, -2)
        rgb_map = torch.zeros((*viewdirs.shape[:-1], 3), device=xyz_sampled.device)
        rgb_map[rays_to_consider], _ = self.renderModule(
            None,
            viewdirs[rays_to_consider],
            cum_app_features[rays_to_consider],
            None,
        )

        if bg_color is None and white_bg:
            bg_color = torch.ones(3, device=rgb_map.device)
        elif bg_color is None:
            bg_color = torch.zeros(3, device=rgb_map.device)

        rgb_map = rgb_map * acc_map[..., None] + bg_color * (1.0 - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]

        return (
            rgb_map,
            depth_map,
            acc_map,
            alpha,
            z_vals,
            dists,
        )  # rgb, sigma, alpha, weight, bg_weight
