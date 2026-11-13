"""Loading pretrained 4DGS models, rendering with watermark offsets, and image metrics."""

import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from arguments import ModelHiddenParams, ModelParams, PipelineParams
from gaussian_renderer import render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.params_utils import merge_hparams


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_config(path: str):
    try:
        from mmcv import Config  # mmcv < 2.0, as in the official 4DGS repository
    except ImportError:
        from mmengine import Config
    return Config.fromfile(path)


def load_pretrained_4dgs(model_path: str, configs: str, iteration: int, source_path: str = None):
    """Load a 4DGS model trained with the official pipeline (`train.py`).

    Returns (gaussians, pipe, background, source_path).
    """
    parser = ArgumentParser()
    model_params, pipe_params, hidden_params = ModelParams(parser), PipelineParams(parser), ModelHiddenParams(parser)
    args = parser.parse_args([])

    cfg_args = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_args):
        with open(cfg_args) as f:
            saved = eval(f.read(), {"Namespace": Namespace})
        for k, v in vars(saved).items():
            setattr(args, k, v)
    if configs:
        args = merge_hparams(args, _load_config(configs))
    if source_path:
        args.source_path = source_path

    gaussians = GaussianModel(args.sh_degree, hidden_params.extract(args))
    ckpt_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    gaussians.load_ply(os.path.join(ckpt_dir, "point_cloud.ply"))
    gaussians.load_model(ckpt_dir)
    # The pretrained 4DGS stays frozen; only the watermark offsets are optimized.
    for p in gaussians._deformation.parameters():
        p.requires_grad_(False)
    for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        getattr(gaussians, name).requires_grad_(False)

    background = torch.tensor([1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0], device="cuda")
    return gaussians, pipe_params.extract(args), background, args.source_path


def render_frame(cam, gaussians, pipe, background, xyz_offset=None, sh_offset=None):
    out = render(cam, gaussians, pipe, background, xyz_offset=xyz_offset, sh_offset=sh_offset)["render"]
    return out.clamp(0.0, 1.0)


def render_clip(cams, gaussians, pipe, background, fourier=None, sh=None):
    """Render a sequence of cameras, optionally with the watermark offsets eps_i(t) (`fourier`) and
    delta_i(t) (`sh`). Returns (T, 3, H, W)."""
    frames = []
    for c in cams:
        t = torch.tensor([c.time], device="cuda")
        xyz_offset = fourier(t)[0] if fourier is not None else None
        sh_offset = sh(t)[0] if sh is not None else None
        frames.append(render_frame(c, gaussians, pipe, background, xyz_offset, sh_offset))
    return torch.stack(frames)


class ImageMetrics:
    """PSNR / SSIM / LPIPS(alex) between rendered and reference frames."""

    def __init__(self):
        self.lpips = LPIPS("alex", "0.1").cuda().eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> dict:
        vals = {"psnr": [], "ssim": [], "lpips": []}
        for p, g in zip(pred, gt):
            vals["psnr"].append(psnr(p[None], g[None]).mean().item())
            vals["ssim"].append(ssim(p[None], g[None]).item())
            vals["lpips"].append(self.lpips(p[None], g[None]).item())
        return {k: float(np.mean(v)) for k, v in vals.items()}
