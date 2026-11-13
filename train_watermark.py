"""Stage 2: embed a watermark message into a pretrained 4DGS model (Sec. 4.3-4.5).

Only the trajectory-aligned positional offsets eps_i(t) and the SH offsets delta_i(t), both
Fourier series over time (Eq. 15), are optimized; the pretrained 4DGS and the Stage-1 decoder stay frozen.

Example (D-NeRF / mutant, 48 bits):
    python train_watermark.py --dataset dnerf \
        --model_path output/dnerf/mutant --configs arguments/dnerf/mutant.py --iteration 20000 \
        --decoder checkpoints/decoder/decoder_48bit.pth --out_dir output/mark4d/mutant_48bit \
        --num_iterations <N> --num_freqs <K> --lr_fourier <LR> --lr_sh <LR> \
        --lambda_recon <W> --lambda_align <W> --lambda_min <W> --lambda_max <W> --max_combine <M>
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import trange

from mark4d.cameras import make_camera_sequences
from mark4d.decoder import bit_accuracy, load_decoder
from mark4d.distortions import RandomDistortion
from mark4d.evaluation import evaluate_watermark, format_results
from mark4d.motion import motion_adaptive_weights, motion_tangent, trajectory_alignment_loss
from mark4d.offsets import FourierOffset, SHOffset
from mark4d.utils import ImageMetrics, load_pretrained_4dgs, render_clip, seed_everything


def pick_message(args, decoder_ckpt) -> torch.Tensor:
    msg_len = decoder_ckpt["msg_len"]
    if args.message:
        bits = [int(b) for b in args.message.strip()]
        if len(bits) != msg_len or set(bits) - {0, 1}:
            raise ValueError(f"--message must be a {msg_len}-bit binary string.")
        return torch.tensor(bits, dtype=torch.float32)
    # Default: a message the decoder never saw during Stage-1 training.
    pool = decoder_ckpt["test_messages"]
    return pool[random.Random(args.seed).randrange(len(pool))].float()


def save_offsets(path, fourier, sh, message, args):
    torch.save(
        {
            "fourier": fourier.state_dict(),
            "sh": sh.state_dict(),
            "num_freqs": fourier.num_freqs,
            "num_sh_coeffs": sh.a.shape[2],
            "message": message.cpu(),
            "args": vars(args),
        },
        path,
    )


def main(args):
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ----------------------------------------------------------- setup
    decoder, decoder_ckpt = load_decoder(args.decoder)
    message = pick_message(args, decoder_ckpt).cuda()
    print(f"Embedding {len(message)}-bit message: {''.join(str(int(b)) for b in message.tolist())}")

    gaussians, pipe, background, source_path = load_pretrained_4dgs(
        args.model_path, args.configs, args.iteration, args.source_path
    )
    print(f"Loaded 4DGS with {gaussians.get_xyz.shape[0]} Gaussians from {args.model_path}")

    train_cams = make_camera_sequences(source_path, args.dataset, args.num_train_cameras, args.num_frames, randomize=True)
    test_cams = make_camera_sequences(source_path, args.dataset, 1, args.num_frames, randomize=False)[0]

    T = args.clip_len
    window_starts = list(range(0, args.num_frames - T + 1, T))
    frame_times = torch.linspace(0, 1, args.num_frames)
    dt = 1.0 / (args.num_frames - 1)

    # Motion-adaptive message-loss weights lambda_{S_t} (Eqs. 16-18), fixed for the scene.
    if args.uniform_weight:
        lambda_table = {s: args.lambda_max for s in window_starts}
    else:
        lambda_table = motion_adaptive_weights(
            gaussians, frame_times, window_starts, T, args.lambda_min, args.lambda_max
        )

    num_points = gaussians.get_xyz.shape[0]
    fourier = FourierOffset(num_points, args.num_freqs).cuda()
    sh = SHOffset(num_points, gaussians.get_features.shape[1], args.num_freqs).cuda()
    optimizer = optim.Adam(
        [
            {"params": fourier.parameters(), "lr": args.lr_fourier},
            {"params": sh.parameters(), "lr": args.lr_sh},
        ],
        eps=1e-15,
    )
    distortion = RandomDistortion(args.distortions, args.max_combine) if args.max_combine > 0 else None
    metrics = ImageMetrics()

    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ----------------------------------------------------------- training
    windows = []
    pbar = trange(1, args.num_iterations + 1, desc="Mark4D")
    for it in pbar:
        if not windows:
            windows = [(c, s) for c in range(len(train_cams)) for s in window_starts]
            random.shuffle(windows)
        cam_idx, start = windows.pop()
        cams = train_cams[cam_idx][start : start + T]
        times = torch.tensor([c.time for c in cams], device="cuda")

        with torch.no_grad():
            gts = render_clip(cams, gaussians, pipe, background)
        frames = render_clip(cams, gaussians, pipe, background, fourier=fourier, sh=sh)

        # Message loss on the (distorted) clip, Eqs. (10), (19)
        clip = frames
        if distortion is not None:
            clip, _ = distortion(frames)
        logits = decoder(clip)
        loss_msg = F.binary_cross_entropy_with_logits(logits, message[None])

        # Reconstruction loss, Eq. (20): mean over frames of L1 + LPIPS
        loss_recon = (frames - gts).abs().mean() + metrics.lpips(frames, gts).squeeze() / T

        # Trajectory alignment loss, Eq. (14)
        if args.lambda_align > 0:
            tangents = torch.stack([motion_tangent(gaussians, t.item(), dt) for t in times])
            loss_align = trajectory_alignment_loss(fourier(times), tangents)
        else:
            loss_align = torch.zeros((), device="cuda")

        lam = lambda_table[start]
        loss = lam * loss_msg + args.lambda_recon * loss_recon + args.lambda_align * loss_align  # Eq. (21)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_description(
            f"[{it}] msg {loss_msg.item():.4f} (lambda {lam:.2f}) | recon {loss_recon.item():.4f} | "
            f"align {loss_align.item():.4f} | bit acc {bit_accuracy(logits, message[None]):.1f}"
        )

        if it % args.eval_interval == 0 or it == args.num_iterations:
            results = evaluate_watermark(
                test_cams, gaussians, pipe, background, decoder, message, fourier, sh, metrics, clip_len=T
            )
            print(f"\n[ITER {it}] " + format_results(results))
            save_offsets(os.path.join(args.out_dir, "offsets.pth"), fourier, sh, message, args)

    print(f"Saved watermark offsets to {os.path.join(args.out_dir, 'offsets.pth')}")


def get_parser():
    p = argparse.ArgumentParser(description="Mark4D Stage 2: watermark embedding into a pretrained 4DGS")
    # data / models
    p.add_argument("--dataset", choices=["dnerf", "dynerf"], required=True)
    p.add_argument("--model_path", required=True, help="pretrained 4DGS output directory")
    p.add_argument("--configs", required=True, help="4DGS config, e.g. arguments/dnerf/mutant.py")
    p.add_argument("--iteration", type=int, required=True, help="4DGS checkpoint (20000 for D-NeRF, 14000 for DyNeRF)")
    p.add_argument("--source_path", default=None, help="dataset path (defaults to the one stored in cfg_args)")
    p.add_argument("--decoder", required=True, help="Stage-1 checkpoint from train_decoder.py")
    p.add_argument("--message", default=None, help="L-bit binary string; default: a held-out decoder test message")
    p.add_argument("--out_dir", required=True)
    # protocol (Sec. 5.1)
    p.add_argument("--num_frames", type=int, default=300)
    p.add_argument("--num_train_cameras", type=int, default=20)
    p.add_argument("--clip_len", type=int, default=8, help="T, frames per clip (X-CLIP base-patch32 expects 8)")
    # optimization hyperparameters (no defaults: set them for your scene)
    p.add_argument("--num_iterations", type=int, required=True)
    p.add_argument("--num_freqs", type=int, required=True, help="K in Eq. (15), used for eps_i(t) and delta_i(t)")
    p.add_argument("--lr_fourier", type=float, required=True, help="learning rate of eps_i(t)")
    p.add_argument("--lr_sh", type=float, required=True, help="learning rate of delta_i(t)")
    p.add_argument("--lambda_recon", type=float, required=True, help="weight of L_recon, Eq. (21)")
    p.add_argument("--lambda_align", type=float, required=True, help="weight of L_align, Eq. (21)")
    p.add_argument("--lambda_min", type=float, required=True, help="Eq. (18)")
    p.add_argument("--lambda_max", type=float, required=True, help="Eq. (18)")
    p.add_argument("--uniform_weight", action="store_true", help="ablation: constant lambda_max for every window")
    # distortion module (Sec. 4.5)
    p.add_argument("--distortions", nargs="+", default=None, help="distortion pool (default: all)")
    p.add_argument("--max_combine", type=int, required=True, help="max distortions combined per clip (0 = none)")
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
