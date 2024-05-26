import torch
from torchvision import transforms
from typing import Sequence
from pose_estimation.ray_preprocessor import RayPreprocessor
from pose_estimation.backbone import create_backbone
from pose_estimation.multihead_attention import MultiHeadAttention

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

class IdentificationModule(torch.nn.Module):
    def __init__(self, backbone_type: str = "superpoint"):
        super().__init__()

        assert backbone_type in ["dino", "superpoint"]

        self.image_preprocessing_net, backbone_wh, img_num_features = create_backbone(
            type=backbone_type, pretrained=True
        )
        self.norm_mean = torch.nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
            requires_grad=False,
        )
        self.norm_std = torch.nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
            requires_grad=False,
        )

        resize_size = 256
        crop_size = 224
        interpolation = transforms.InterpolationMode.BICUBIC
        transforms_list = [
            transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
            transforms.CenterCrop(crop_size),
            make_normalize_transform(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
        self.transformations = transforms.Compose(transforms_list)
        self.mask_transformations = transforms.Compose(
            [
                transforms.Resize(
                    resize_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(crop_size),
                transforms.Resize(
                    backbone_wh[0],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            ]
        )

        self.backbone_wh = backbone_wh
        self.img_num_features = img_num_features

        self.ray_preprocessor = RayPreprocessor(
            featureC=256, fea_output=img_num_features
        )
        ray_fea_size = img_num_features
        img_fea_size = img_num_features + 14

        self.attention = MultiHeadAttention(
            ray_fea_size, img_fea_size, img_num_features, 1
        )

    @staticmethod
    def get_img_position_encoding(
        img_features_shape, freqs, dtype=torch.float32, device="cpu"
    ):
        meshgrid_elements = []
        for size in img_features_shape:
            meshgrid_elements.append(
                torch.linspace(-1.0, 1.0, steps=size, dtype=dtype, device=device)
            )
        positions = torch.stack(
            torch.meshgrid(*meshgrid_elements, indexing="ij"), dim=-1
        )
        positions = positions.reshape(-1, positions.shape[-1])
        # start computing the positional encoding itself
        freq_bands = (2 ** torch.arange(freqs).float()).to(
            positions.device, non_blocking=True
        )  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1],)
        )  # (..., DF)
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1)
        # back to the original size
        pts = pts.reshape(*img_features_shape, pts.shape[-1])
        return pts

    def compute_features(
        self,
        img: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        rays_rgb: torch.Tensor,
        rays_to_test: int = -1,
    ):
        features_img_w_pe_flat, _ = self.image_processing(img)

        # features_img_w_pe_flat.register_hook(
        #     lambda grad: print("features_img:", grad.max(), grad.min())
        # )

        #

        # ray random selection
        used_ray_ids = torch.randperm(
            rays_ori.shape[0], device=img.device, dtype=torch.long
        )
        if rays_to_test != -1:
            used_ray_ids = used_ray_ids[:rays_to_test]
        # ray processing
        features_rays = self.ray_preprocessor(
            rays_ori[used_ray_ids], rays_dir[used_ray_ids], rays_rgb[used_ray_ids]
        )

        return features_img_w_pe_flat, features_rays, used_ray_ids

    def image_processing(self, img, mask):
        # image processing
        permuted_img = img[None].permute(0, 3, 1, 2)  # [B, H, W, 3] => [B, 3, H, W]

        norm_img = self.transformations(permuted_img)
        mask_img = self.mask_transformations(mask[None, None] * 1.0)[0, 0] > 0.1

        img_features = self.image_preprocessing_net.forward_features(norm_img)[
            "x_norm_patchtokens"
        ][0]
        img_features = img_features.reshape(
            self.backbone_wh[0], self.backbone_wh[1], self.img_num_features
        )
        img_features_np_like = img_features
        img_features = img_features.permute(2, 0, 1)
        # img_features = self.image_preprocessing_net(norm_img)[
        #     0
        # ]  # [B, C, H // 2, W // 2]

        position_encoding = self.get_img_position_encoding(
            img_features.shape[-2:], 3, dtype=img.dtype, device=img.device
        )
        features_img_w_pe = torch.cat(
            [img_features, position_encoding.permute(2, 0, 1)], dim=0
        )
        # features_img_w_pe = img_features
        features_img_w_pe = features_img_w_pe.permute(1, 2, 0)
        return (
            features_img_w_pe[mask_img].view(-1, features_img_w_pe.shape[-1]),
            img_features_np_like[mask_img].view(-1, img_features_np_like.shape[-1]),
        )

    def run_attention(self, img, mask, rays_ori, rays_dir, rays_rgb):
        features_img_w_pe_flat, features_img_flat = self.image_processing(img, mask)
        features_rays = self.ray_preprocessor(rays_ori, rays_dir, rays_rgb)
        attention_map = self.attention(features_img_w_pe_flat, features_rays, mask=None)
        # score = 1.0 - torch.prod(1.0 - attention_map, dim=0)
        score = torch.sum(attention_map, dim=0)
        return score, attention_map, features_img_flat

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        rays_rgb: torch.Tensor,
        rays_to_test: int = -1,
    ):
        used_ray_ids = torch.randperm(
            rays_ori.shape[0], device=img.device, dtype=torch.long
        )
        if rays_to_test != -1:
            used_ray_ids = used_ray_ids[:rays_to_test]
        scores, attention_map, features_img_w_pe_flat = self.run_attention(
            img,
            mask,
            rays_ori[used_ray_ids],
            rays_dir[used_ray_ids],
            rays_rgb[used_ray_ids],
        )
        return scores, attention_map, features_img_w_pe_flat, used_ray_ids

    @torch.no_grad()
    def test_image(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        rays_rgb: torch.Tensor,
        rays_to_output: int = 100,
    ):
        scores, attention_map, _ = self.run_attention(
            img, mask, rays_ori, rays_dir, rays_rgb
        )

        chunk_topk = torch.topk(scores, k=rays_to_output)

        return chunk_topk.indices, chunk_topk.values, scores, attention_map
