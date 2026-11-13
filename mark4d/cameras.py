"""Fixed-viewpoint camera sequences for watermarking (Sec. 5.1).

Following CopyRNeRF / GuardSplat, we place 21 cameras around each scene and render 300
consecutive frames from every camera with the pretrained 4DGS model; 20 cameras are used for
watermark training and one is held out for evaluation. D-NeRF cameras lie on a circle around
the object (Blender convention); DyNeRF cameras lie on a spiral around the average LLFF pose.

Adapted from GuardSplat (https://narcissusex.github.io/GuardSplat/) and CopyRNeRF
(https://github.com/luo-ziyuan/CopyRNeRF-code).
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from scene.cameras import MiniCam
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix

ZNEAR, ZFAR = 0.01, 100.0


def _normalize(x):
    return x / torch.linalg.norm(x)


def _viewmatrix(z, up, pos):
    vec2 = _normalize(z)
    vec0 = _normalize(torch.linalg.cross(up, vec2))
    vec1 = _normalize(torch.linalg.cross(vec2, vec0))
    return torch.stack([vec0, vec1, vec2, pos], 1)


def _poses_avg(c2ws):
    center = c2ws[:, :3, 3].mean(0)
    vec2 = _normalize(c2ws[:, :3, 2].sum(0))
    up = c2ws[:, :3, 1].sum(0)
    return _viewmatrix(vec2, up, center)


def _se3(r, t):
    m = torch.eye(4, device=r.device, dtype=r.dtype).repeat(len(r), 1, 1)
    m[:, :3, :3] = r
    m[:, :3, 3] = t
    return m


def _recenter_poses(c2ws):
    out = c2ws.clone()
    bottom = torch.tensor([0, 0, 0, 1], device=c2ws.device, dtype=c2ws.dtype)[None]
    c2w = torch.cat([_poses_avg(c2ws), bottom])
    poses = torch.cat([c2ws[:, :3, :4], bottom.repeat(len(c2ws), 1, 1)], 1)
    poses = torch.linalg.inv(c2w).repeat(len(poses), 1, 1).bmm(poses)
    out[:, :3, :4] = poses[:, :3, :4]
    return out


def _pose_spherical(theta, phi, radius):
    trans = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, radius], [0, 0, 0, 1]], dtype=torch.float32)
    p = phi / 180.0 * np.pi
    rot_phi = torch.tensor(
        [[1, 0, 0, 0], [0, np.cos(p), -np.sin(p), 0], [0, np.sin(p), np.cos(p), 0], [0, 0, 0, 1]], dtype=torch.float32
    )
    th = theta / 180.0 * np.pi
    rot_theta = torch.tensor(
        [[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]], dtype=torch.float32
    )
    flip = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
    return flip @ rot_theta @ rot_phi @ trans


def scene_camera_info(source_path: str, dataset: str) -> dict:
    """Intrinsics and the reference frame used to place the virtual cameras."""
    if dataset == "dynerf":
        extr = sorted(read_extrinsics_binary(os.path.join(source_path, "colmap/sparse/0/images.bin")).values(), key=lambda c: c.id)
        intr = read_intrinsics_binary(os.path.join(source_path, "colmap/sparse/0/cameras.bin"))[1]
        width, height, focal_x = intr.width, intr.height, intr.params[0]
        fovx, fovy = focal2fov(focal_x, width), focal2fov(focal_x, height)

        rs = torch.tensor(np.stack([qvec2rotmat(c.qvec) for c in extr]), dtype=torch.float32).cuda()
        ts = torch.tensor(np.stack([c.tvec for c in extr]), dtype=torch.float32).cuda()
        c2ws = torch.linalg.inv(_se3(rs, ts))

        bds = np.load(os.path.join(source_path, "poses_bounds.npy"))[:, -2:].astype(np.float32)
        sc = 1.0 / (bds.min() * 0.75)
        bds = bds * sc
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        c2ws[:, :3, 3] *= sc
        c2ws = _recenter_poses(c2ws)

        rads = torch.ones(4, device="cuda")
        rads[:3] = torch.quantile(torch.abs(c2ws[:, :3, 3]), 0.9, 0)
        dt = 0.75
        return dict(
            width=width, height=height, fovx=fovx, fovy=fovy,
            c2w=_poses_avg(c2ws), up=_normalize(c2ws[:, :3, 1].sum(0)), rads=rads,
            focal=1.0 / ((1.0 - dt) / close_depth + dt / inf_depth),
        )

    if dataset == "dnerf":
        with open(os.path.join(source_path, "transforms_train.json")) as f:
            contents = json.load(f)
        fovx = contents["camera_angle_x"]
        with Image.open(os.path.join(source_path, contents["frames"][0]["file_path"] + ".png")) as im:
            width, height = im.size
        return dict(width=width, height=height, fovx=fovx, fovy=focal2fov(fov2focal(fovx, width), height))

    raise ValueError(f"Unknown dataset type: {dataset}")


def _blender_w2cs(num_cameras, randomize, theta_min=90.0, theta_max=270.0, phi=-30.0, radius=4.031128874):
    if randomize:
        thetas = np.random.uniform(theta_min, theta_max, num_cameras)
    elif num_cameras == 1:
        thetas = np.array([(theta_min + theta_max) / 2.0])
    else:
        thetas = np.linspace(theta_min, theta_max, num_cameras)
    c2ws = torch.stack([_pose_spherical(t, phi, radius) for t in thetas], 0)
    c2ws[:, :3, 1:3] *= -1
    return torch.stack([torch.linalg.inv(c2w).transpose(0, 1) for c2w in c2ws]).cuda()


def _llff_w2cs(num_cameras, randomize, info, rots=2, zrate=0.5, rad_scale=0.5):
    c2w = info["c2w"].repeat(num_cameras, 1, 1)
    rads = info["rads"] * rad_scale
    if randomize:
        thetas = torch.rand(num_cameras) * 2 * np.pi * rots
    else:
        thetas = torch.linspace(0.0, 2 * np.pi * rots, num_cameras + 1)[:-1]
    xs = torch.stack(
        [torch.tensor([np.cos(th), -np.sin(th), -np.sin(th * zrate), 1.0], dtype=torch.float32).cuda() * rads for th in thetas]
    )
    cs = torch.bmm(c2w, xs[..., None]).squeeze(-1)
    fs = torch.tensor([0, 0, -info["focal"], 1], dtype=torch.float32).repeat(num_cameras, 1)[..., None].cuda()
    zs = cs - torch.bmm(c2w, fs).squeeze(-1)
    m34 = torch.stack([_viewmatrix(_normalize(z), info["up"], c) for z, c in zip(zs, cs)])
    poses = _se3(m34[:, :3, :3], m34[:, :3, 3])
    # world-to-view matrices (same convention as GuardSplat)
    return torch.stack([torch.linalg.inv(p) for p in poses])


def make_camera_sequences(source_path: str, dataset: str, num_cameras: int, num_frames: int, randomize: bool):
    """Returns `num_cameras` lists of `num_frames` MiniCams with times linspace(0, 1, num_frames)."""
    info = scene_camera_info(source_path, dataset)
    w2cs = _blender_w2cs(num_cameras, randomize) if dataset == "dnerf" else _llff_w2cs(num_cameras, randomize, info)
    proj = getProjectionMatrix(znear=ZNEAR, zfar=ZFAR, fovX=info["fovx"], fovY=info["fovy"]).transpose(0, 1).cuda()
    times = torch.linspace(0, 1, num_frames).tolist() if num_frames > 1 else [0.0]
    sequences = []
    for w2c in w2cs:
        full = w2c @ proj
        sequences.append(
            [MiniCam(info["width"], info["height"], info["fovy"], info["fovx"], ZNEAR, ZFAR, w2c, full, t) for t in times]
        )
    return sequences
