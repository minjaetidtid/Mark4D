"""Evaluation protocol (Sec. 5.1).

* Video-level bit accuracy: the held-out test sequence is split into non-overlapping
  T-frame clips and every clip is decoded by D_M(E_V(.)).
* Frame-level bit accuracy: every frame is decoded on its own as a static clip built by
  repeating the frame T times (for comparison with frame-level baselines).
* Invisibility: PSNR / SSIM / LPIPS between watermarked and original renderings.
* Robustness: the same accuracies after real (non-STE) frame-level / video-level distortions
  and after model-level distortion of the Gaussians.
"""

from contextlib import contextmanager

import torch

from mark4d.decoder import bit_accuracy
from mark4d.distortions import FRAME_LEVEL, VIDEO_LEVEL, apply_distortion
from mark4d.utils import render_clip


@torch.no_grad()
def video_bit_accuracy(frames, decoder, message, clip_len, distortion=None):
    accs = []
    for s in range(0, len(frames) - clip_len + 1, clip_len):
        clip = apply_distortion(frames[s : s + clip_len].cuda(), distortion)
        accs.append(bit_accuracy(decoder(clip), message[None]))
    return sum(accs) / len(accs)


@torch.no_grad()
def frame_bit_accuracy(frames, decoder, message, clip_len, distortion=None):
    accs = []
    for f in frames:
        f = apply_distortion(f[None].cuda(), distortion)
        accs.append(bit_accuracy(decoder(f.repeat(clip_len, 1, 1, 1)), message[None]))
    return sum(accs) / len(accs)


@contextmanager
def distorted_gaussians(gaussians, sigma=0.01, remove_ratio=0.2, seed=0):
    """Model-level distortion: Gaussian noise on the Gaussian positions and random removal
    of a fraction of the Gaussians (implemented by zeroing their opacity)."""
    xyz, opacity = gaussians._xyz.data.clone(), gaussians._opacity.data.clone()
    g = torch.Generator(device=xyz.device).manual_seed(seed)
    gaussians._xyz.data += sigma * torch.randn(xyz.shape, generator=g, device=xyz.device)
    removed = torch.rand(xyz.shape[0], generator=g, device=xyz.device) < remove_ratio
    gaussians._opacity.data[removed] = -1e4
    try:
        yield
    finally:
        gaussians._xyz.data.copy_(xyz)
        gaussians._opacity.data.copy_(opacity)


@torch.no_grad()
def evaluate_watermark(
    cams, gaussians, pipe, background, decoder, message, fourier, sh, metrics, clip_len=8,
    distortions=(), model_level=False, return_frames=False,
):
    gts = render_clip(cams, gaussians, pipe, background).cpu()
    frames = render_clip(cams, gaussians, pipe, background, fourier=fourier, sh=sh).cpu()

    results = {
        "bit_acc_video": video_bit_accuracy(frames, decoder, message, clip_len),
        "bit_acc_frame": frame_bit_accuracy(frames, decoder, message, clip_len),
        **metrics(frames.cuda(), gts.cuda()),
    }
    for name in distortions:
        results[f"{name}/bit_acc_video"] = video_bit_accuracy(frames, decoder, message, clip_len, name)
        if name in FRAME_LEVEL:
            results[f"{name}/bit_acc_frame"] = frame_bit_accuracy(frames, decoder, message, clip_len, name)
        elif name not in VIDEO_LEVEL:
            raise ValueError(f"Unknown distortion: {name}")
    if model_level:
        with distorted_gaussians(gaussians):
            distorted = render_clip(cams, gaussians, pipe, background, fourier=fourier, sh=sh).cpu()
        results["model/bit_acc_video"] = video_bit_accuracy(distorted, decoder, message, clip_len)
        results["model/bit_acc_frame"] = frame_bit_accuracy(distorted, decoder, message, clip_len)
    return (results, frames, gts) if return_frames else results


def format_results(results: dict) -> str:
    return " | ".join(f"{k} {v:.4f}" if "/" not in k and k in ("ssim", "lpips") else f"{k} {v:.2f}" for k, v in results.items())
