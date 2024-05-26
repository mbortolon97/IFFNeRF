from typing import Optional

import torch


def linear_to_srgb(linear: torch.Tensor,
                   eps: Optional[float] = None) -> torch.Tensor:
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(linear.dtype).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb: torch.Tensor,
                   eps: Optional[float] = None) -> torch.Tensor:
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(srgb.dtype).eps
    linear0 = 25 / 323 * srgb
    linear1 = torch.clamp(((200 * srgb + 11) / 211), min=False) ** (12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)


# FYI: converted for future use, but in theory should not be required
def color_correct(img: torch.Tensor, ref: torch.Tensor, num_iters: int = 5, eps: float = 0.5 / 255):
    """Warp `img` to match the colors in `ref_img`."""
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = torch.cat(a_mat, dim=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = torch.where(mask[:, None], a_mat, 0)
            mb = torch.where(mask, b, 0)
            # Solve the linear system. We're using the np.lstsq instead of jnp because
            # it's significantly more stable in this case, for some reason.
            w = torch.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert torch.all(torch.isfinite(w))
            warp.append(w)
        warp = torch.stack(warp, dim=-1)
        # Apply the warp to update img_mat.
        img_mat = torch.clip(
            torch.matmul(a_mat, warp), 0, 1)
    corrected_img = torch.reshape(img_mat, img.shape)
    return corrected_img
