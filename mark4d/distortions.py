"""Differentiable distortion module A (Sec. 4.5, Supp. A.2).

Frame-level: Gaussian noise, rotation, scaling, Gaussian blur, cropping, brightness, JPEG.
Video-level: H.264 compression, random frame swap.

Non-differentiable operations (JPEG, H.264, frame swap) use a straight-through estimator
during training: I + stopgrad(A(I) - I). With `ste=False` the real operation is applied,
which is what we use for evaluation.

All functions take a clip of shape (T, 3, H, W) with values in [0, 1].
"""

import io
import random

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _ste(clip: torch.Tensor, out: torch.Tensor, ste: bool) -> torch.Tensor:
    return clip + (out - clip).detach() if ste else out


# ----------------------------- frame-level ---------------------------------
def gaussian_noise(clip, sigma=0.1, **_):
    return clip + sigma * torch.randn_like(clip)


def rotation(clip, max_angle=30.0, **_):
    angle = random.uniform(-max_angle, max_angle)  # [-pi/6, +pi/6], shared by all frames
    return TF.rotate(clip, angle, interpolation=InterpolationMode.BILINEAR)


def scaling(clip, min_ratio=0.75, max_ratio=1.25, **_):
    ratio = random.uniform(min_ratio, max_ratio)
    return F.interpolate(clip, scale_factor=ratio, mode="bilinear", align_corners=False)


def gaussian_blur(clip, kernel_size=3, sigma=0.1, **_):
    return TF.gaussian_blur(clip, kernel_size=kernel_size, sigma=sigma)


def crop(clip, ratio=0.4, **_):
    h, w = clip.shape[-2:]
    return TF.center_crop(clip, [max(1, int(ratio * h)), max(1, int(ratio * w))])


def brightness(clip, min_factor=0.5, max_factor=1.5, **_):
    return TF.adjust_brightness(clip, random.uniform(min_factor, max_factor))


def jpeg(clip, quality=50, ste=True, **_):
    with torch.no_grad():
        frames = (clip.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        out = []
        for f in frames:
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            dec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            out.append(torch.from_numpy(dec))
        out = torch.stack(out).to(clip.device).permute(0, 3, 1, 2).float().div(255)
    return _ste(clip, out, ste)


# ----------------------------- video-level ---------------------------------
def h264(clip, crf=30, fps=30, ste=True, **_):
    with torch.no_grad():
        t, _, h, w = clip.shape
        h2, w2 = h - h % 2, w - w % 2  # yuv420p needs even sizes
        frames = (clip[..., :h2, :w2].clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        buf = io.BytesIO()
        with av.open(buf, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = w2, h2, "yuv420p"
            stream.options = {"crf": str(crf)}
            for f in frames:
                for pkt in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
        buf.seek(0)
        with av.open(buf, "r") as container:
            dec = [torch.from_numpy(fr.to_ndarray(format="rgb24")) for fr in container.decode(video=0)]
        out = torch.stack(dec[:t]).to(clip.device).permute(0, 3, 1, 2).float().div(255)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False) if (h2, w2) != (h, w) else out
    return _ste(clip, out, ste)


def frame_swap(clip, ratio=0.25, ste=True, **_):
    """Randomly exchange `ratio` of the frames of the clip (no frame stays in place)."""
    t = clip.shape[0]
    k = max(2, int(round(ratio * t)))
    idx = random.sample(range(t), k)
    perm = idx[1:] + idx[:1]  # cyclic shift -> a derangement of the selected frames
    out = clip.clone()
    out[idx] = clip[perm]
    return _ste(clip, out.detach(), ste)


FRAME_LEVEL = {
    "noise": gaussian_noise,
    "rotation": rotation,
    "scaling": scaling,
    "blur": gaussian_blur,
    "crop": crop,
    "brightness": brightness,
    "jpeg": jpeg,
}
VIDEO_LEVEL = {
    "h264": h264,
    "frame_swap": frame_swap,
}
ALL_DISTORTIONS = {**FRAME_LEVEL, **VIDEO_LEVEL}


def apply_distortion(clip: torch.Tensor, name: str, ste: bool = False) -> torch.Tensor:
    if name in (None, "none", "clean"):
        return clip
    return ALL_DISTORTIONS[name](clip, ste=ste)


class RandomDistortion:
    """Training-time distortion: a random combination of distortions from `pool`.

    For every clip we draw k ~ U{0, ..., max_combine} distinct distortions and apply them in
    order (frame-level before video-level). k = 0 leaves the clip clean.
    """

    def __init__(self, pool, max_combine: int):
        self.pool = list(pool) if pool else list(ALL_DISTORTIONS)
        unknown = set(self.pool) - set(ALL_DISTORTIONS)
        if unknown:
            raise ValueError(f"Unknown distortions: {sorted(unknown)}")
        self.max_combine = max_combine

    def __call__(self, clip: torch.Tensor):
        k = random.randint(0, min(self.max_combine, len(self.pool)))
        names = random.sample(self.pool, k)
        names.sort(key=lambda n: n in VIDEO_LEVEL)
        for name in names:
            clip = ALL_DISTORTIONS[name](clip, ste=True)
        return clip, names
