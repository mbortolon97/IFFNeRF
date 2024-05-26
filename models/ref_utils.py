import math

import torch


def reflect(viewdirs: torch.Tensor, normals: torch.Tensor):
    """
    Reflect view directions about normals.
    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.
    Args:
        viewdirs: [..., 3] array of view directions.
        normals: [..., 3] array of normal directions (assumed to be unit vectors).
    Returns:
        [..., 3] array of reflection directions.
    """
    return torch.multiply(2.0 * torch.bmm(normals.view(-1, 1, 3), viewdirs.view(-1, 3, 1))[..., 0], normals) - viewdirs
    # assert torch.allclose(result_viewdir[:10], 2.0 * torch.sum(normals[:10] * viewdirs[:10, ..., None, :], dim=-1,
    # keepdim=True) * normals[:10] - viewdirs[:10, ..., None, :]), 'values not equal'


class IntegratedDirEnc(torch.nn.Module):
    @staticmethod
    def generalized_binomial_coeff(a, k):
        """Compute generalized binomial coefficients."""
        return torch.prod(a - torch.arange(k)) / math.factorial(k)

    @staticmethod
    def assoc_legendre_coeff(l, m, k):
        """
        Compute associated Legendre polynomial coefficients.
        Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
        (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).
        Args:
            l: associated Legendre polynomial degree.
            m: associated Legendre polynomial order.
            k: power of cos(theta).
        Returns:
            A float, the coefficient of the term corresponding to the inputs.
        """
        return ((-1) ** m * 2 ** l * math.factorial(l) / math.factorial(k) /
                math.factorial(l - k - m) *
                IntegratedDirEnc.generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))

    @staticmethod
    def sph_harm_coeff(l, m, k):
        """Compute spherical harmonic coefficients."""
        return (math.sqrt(
            (2.0 * l + 1.0) * math.factorial(l - m) /
            (4.0 * math.pi * math.factorial(l + m))) * IntegratedDirEnc.assoc_legendre_coeff(l, m, k))

    @staticmethod
    def get_ml_array(deg_view: int):
        """Create a list with all pairs of (l, m) values to use in the encoding."""
        ml_list = []
        for i in range(deg_view):
            l = 2 ** i
            # Only use nonnegative m values, later splitting real and imaginary parts.
            for m in range(l + 1):
                ml_list.append((m, l))

        # Convert list into a numpy array.
        ml_array = torch.tensor(ml_list).T
        return ml_array

    def __init__(self, deg_view: int):
        super().__init__()

        self.ml_array = torch.nn.Parameter(IntegratedDirEnc.get_ml_array(deg_view), requires_grad=False)
        l_max = 2**(deg_view - 1)

        # Create a matrix corresponding to ml_array holding all coefficients, which,
        # when multiplied (from the right) by the z coordinate Vandermonde matrix,
        # results in the z component of the encoding.
        mat = torch.zeros((l_max + 1, self.ml_array.shape[1]))
        for i, (m, l) in enumerate(self.ml_array.T):
            for k in range(l - m + 1):
                mat[k, i] = IntegratedDirEnc.sph_harm_coeff(l, m, k)
        self.mat = torch.nn.Parameter(mat, requires_grad=False)

    def forward(self, xyz: torch.Tensor, kappa_inv: torch.Tensor):
        """Function returning integrated directional encoding (IDE).
        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.
        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.pow(z, torch.arange(self.mat.shape[0], dtype=z.dtype, device=z.device)[None, :])
        # vmz = torch.concatenate([z ** i for i in range(self.mat.shape[0])], axis=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.pow((x + 1j * y), self.ml_array[0, :])
        # vmxy = jnp.concatenate([(x + 1j * y) ** m for m in ml_array[0, :]], axis=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, self.mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * self.ml_array[1, :] * (self.ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.view_as_real(ide)