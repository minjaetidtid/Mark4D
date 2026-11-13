"""Motion-related terms: trajectory alignment loss (Sec. 4.3) and motion-adaptive
loss weighting (Sec. 4.4). All motion quantities are read from the frozen, pretrained
4DGS deformation field F_theta (Eq. 5)."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def deformed_xyz(gaussians, t: float) -> torch.Tensor:
    """x_i(t): positions of all Gaussians deformed to normalized time t. Returns (N, 3)."""
    xyz = gaussians.get_xyz
    time = torch.full((xyz.shape[0], 1), float(t), device=xyz.device)
    out = gaussians._deformation(
        xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, time
    )
    return out[0]


@torch.no_grad()
def motion_tangent(gaussians, t: float, dt: float) -> torch.Tensor:
    """d_i(t) of Eq. (13): normalized central difference x_i(t+dt) - x_i(t-dt).

    At the ends of the sequence the difference becomes one-sided. Returns (N, 3).
    """
    t0, t1 = max(t - dt, 0.0), min(t + dt, 1.0)
    diff = deformed_xyz(gaussians, t1) - deformed_xyz(gaussians, t0)
    return diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)


def trajectory_alignment_loss(xyz_offsets: torch.Tensor, tangents: torch.Tensor) -> torch.Tensor:
    """L_align of Eq. (14): mean over Gaussians and frames of 1 - cos(eps_i(tau), d_i(tau)).

    xyz_offsets, tangents: (T, N, 3).
    """
    cos = F.cosine_similarity(xyz_offsets, tangents, dim=-1, eps=1e-8)
    return (1.0 - cos).mean()


@torch.no_grad()
def frame_displacements(gaussians, frame_times: torch.Tensor) -> torch.Tensor:
    """mean_i ||x_i(t_{k+1}) - x_i(t_k)|| for consecutive frames k = 0 .. F-2. Returns (F-1,)."""
    prev = deformed_xyz(gaussians, frame_times[0].item())
    disp = []
    for t in frame_times[1:]:
        cur = deformed_xyz(gaussians, t.item())
        disp.append((cur - prev).norm(dim=-1).mean())
        prev = cur
    return torch.stack(disp)


@torch.no_grad()
def motion_adaptive_weights(
    gaussians, frame_times: torch.Tensor, window_starts, clip_len: int, lambda_min: float, lambda_max: float
) -> dict:
    """lambda_{S_t} of Eqs. (16)-(18) for every temporal window S_t.

    Delta_{S_t} averages ||x_i(tau + dt) - x_i(tau)|| over the Gaussians and the frames tau of the
    window (Eq. 16), Delta is min-max normalized over all windows of the scene (Eq. 17), and the
    message-loss weight interpolates linearly between lambda_min and lambda_max (Eq. 18).
    Returns {window_start: lambda}.
    """
    disp = frame_displacements(gaussians, frame_times)  # (F-1,)
    deltas = {}
    for s in window_starts:
        steps = disp[s : min(s + clip_len, len(disp))]  # tau + dt must stay inside the sequence
        deltas[s] = steps.mean().item()
    d_min, d_max = min(deltas.values()), max(deltas.values())
    span = max(d_max - d_min, 1e-12)
    return {s: (1.0 - (d - d_min) / span) * lambda_min + ((d - d_min) / span) * lambda_max for s, d in deltas.items()}
