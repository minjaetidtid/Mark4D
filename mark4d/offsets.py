"""Learnable watermark offsets (Sec. 4.1 and 4.3).

The watermarked 4DGS (Eq. 6) is
    G~(t) = { x_i(t) + eps_i(t),  h_i(t) + delta_i(t),  alpha_i(t),  Sigma_i(t) },
i.e. only the deformed positions and SH coefficients are perturbed; opacity and covariance
are left untouched (Supp. C.2).

Both time-dependent offsets are parameterized as Fourier series over normalized time
t in [0, 1] (Eq. 15):
    f_i(t) = sum_{k=1..K} a_{i,k} sin(2 pi k t) + b_{i,k} cos(2 pi k t).
"""

import math

import torch
from torch import nn


class FourierSeries(nn.Module):
    """Per-Gaussian Fourier series with coefficients of shape (N, K, *feature_shape)."""

    def __init__(self, num_points: int, feature_shape, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        self.a = nn.Parameter(torch.zeros(num_points, num_freqs, *feature_shape))
        self.b = nn.Parameter(torch.zeros(num_points, num_freqs, *feature_shape))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) normalized times -> offsets (B, N, *feature_shape)."""
        k = torch.arange(1, self.num_freqs + 1, device=t.device, dtype=t.dtype)
        phase = 2.0 * math.pi * t[:, None] * k[None]  # (B, K)
        return torch.einsum("bk,nk...->bn...", torch.sin(phase), self.a) + torch.einsum(
            "bk,nk...->bn...", torch.cos(phase), self.b
        )


class FourierOffset(FourierSeries):
    """Trajectory-aligned positional offset eps_i(t) in R^3 (Eq. 15)."""

    def __init__(self, num_points: int, num_freqs: int):
        super().__init__(num_points, (3,), num_freqs)


class SHOffset(FourierSeries):
    """Offset delta_i(t) on the SH coefficients h_i(t), in the layout of
    GaussianModel.get_features: (N, (D+1)^2, 3)."""

    def __init__(self, num_points: int, num_sh_coeffs: int, num_freqs: int):
        super().__init__(num_points, (num_sh_coeffs, 3), num_freqs)
