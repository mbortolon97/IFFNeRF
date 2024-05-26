import torch
import math

import models.ref_utils as ref_utils
import models.image as image
from omegaconf import OmegaConf


class MultiplierModule(torch.nn.Module):
    """
    Stupid module that multiply a given value to the output of a neural network
    """

    def __init__(self, multiply_value: float):
        super().__init__()

        self.multiply_value = multiply_value

    def forward(self, x: torch.Tensor):
        return x * self.multiply_value


class AddModule(torch.nn.Module):
    """
    Stupid module that add a given value to the output of a neural network
    """

    def __init__(self, add_value: float):
        super().__init__()

        self.add_value = add_value

    def forward(self, x: torch.Tensor):
        return x + self.add_value


class NormalizeModule(torch.nn.Module):
    def __init__(self, p: int = 2, dim: int = 1):
        super().__init__()

        self.p = p
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.normalize(x, p=self.p, dim=self.dim)


class Ref(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        viewpe=6,
        feature_c=128,
        deg_view=4,
        predicted_normals=True,
        rgb_premultiplier=1.0,
        rgb_bias=0.0,
    ):
        super(Ref, self).__init__()

        # Precompute and define viewdir or refdir encoding function.
        self.dir_enc_fn = ref_utils.IntegratedDirEnc(deg_view)

        self.rgb_padding: float = 0.001

        self.in_mlpC = (3 + 2 * viewpe * 3) + in_channels
        self.viewpe = viewpe

        self.diffuse_color_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 3),
            AddModule(-math.log(3.0)),
            torch.nn.Sigmoid(),
        )
        self.tint_color_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 3), torch.nn.Sigmoid()
        )
        self.roughness_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 1), AddModule(-1.0), torch.nn.Softplus()
        )

        self.bottleneck_mlp = torch.nn.Linear(in_channels, feature_c)

        self.predicted_normals = predicted_normals
        if self.predicted_normals:
            self.normal_mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, 3),
                NormalizeModule(p=2, dim=-1),
                MultiplierModule(-1),
            )

        specular_mlp_layers = [
            torch.nn.Linear(
                feature_c + sum((2**i) + 1 for i in range(deg_view)) * 2 + 1, 3
            )
        ]
        if rgb_premultiplier < 1.0 - 1e-7 or rgb_premultiplier > 1.0 + 1e-7:
            specular_mlp_layers.append(MultiplierModule(rgb_premultiplier))
        if rgb_bias > 1e-7:
            specular_mlp_layers.append(AddModule(rgb_bias))
        specular_mlp_layers.append(torch.nn.Sigmoid())
        self.specular_mlp = torch.nn.Sequential(*specular_mlp_layers)

    def forward(self, pts, viewdirs, features, normals):
        # Predict diffuse color.
        if normals is None and self.predicted_normals:
            normals = self.normal_mlp(features)

        tint = self.tint_color_mlp(features)
        roughness = self.roughness_mlp(features)

        # Output of the first part of MLP.
        bottleneck = self.bottleneck_mlp(features)

        # Encode view (or reflection) directions.
        # Compute reflection directions. Note that we flip viewdirs before
        # reflecting, because they point from the camera to the point,
        # whereas ref_utils.reflect() assumes they point toward the camera.
        # Returned refdirs then point from the point to the environment.
        refdirs = ref_utils.reflect(-viewdirs, normals)
        # Encode reflection directions.
        dir_enc = self.dir_enc_fn(refdirs, roughness)

        # Append dot product between normal vectors and view directions.
        dotprod = torch.bmm(normals.view(-1, 1, 3), viewdirs.view(-1, 3, 1))[..., 0]

        # Append view (or reflection) direction encoding to bottleneck vector.
        x = [
            bottleneck,
            dir_enc.view(dir_enc.shape[0], math.prod(dir_enc.shape[1:])),
            dotprod,
        ]

        # Concatenate bottleneck, directional encoding, dotprod
        x = torch.cat(x, dim=-1)

        # Output of the specular color
        rgb = self.specular_mlp(x)
        specular_linear = tint * rgb

        # Initialize linear diffuse color around 0.25, so that the combined
        # linear color is initialized around 0.5.
        diffuse_linear = self.diffuse_color_mlp(features)

        # Combine specular and diffuse components and tone map to sRGB.
        rgb = torch.clip(
            image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0
        )

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return rgb, None

    def compute_normals(self, features: torch.Tensor):
        return -self.normal_mlp(features)
