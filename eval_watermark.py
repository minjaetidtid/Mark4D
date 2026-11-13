"""Evaluate a watermarked 4DGS: capacity, invisibility and robustness (Sec. 5.1, Tab. 1-2).

Example:
    python eval_watermark.py --offsets output/mark4d/mutant_48bit/offsets.pth \
        --decoder checkpoints/decoder/decoder_48bit.pth --distortions all --model_level --save_frames
"""

import argparse
import json
import os

import torch
from PIL import Image

from mark4d.cameras import make_camera_sequences
from mark4d.decoder import load_decoder
from mark4d.distortions import ALL_DISTORTIONS
from mark4d.evaluation import evaluate_watermark, format_results
from mark4d.offsets import FourierOffset, SHOffset
from mark4d.utils import ImageMetrics, load_pretrained_4dgs, seed_everything


def main(args):
    seed_everything(args.seed)
    ckpt = torch.load(args.offsets, map_location="cpu")
    train_args = ckpt["args"]

    decoder, _ = load_decoder(args.decoder)
    message = ckpt["message"].cuda()
    if decoder.msg_len != len(message):
        raise ValueError("The decoder and the watermark offsets use different message lengths.")

    gaussians, pipe, background, source_path = load_pretrained_4dgs(
        train_args["model_path"], train_args["configs"], train_args["iteration"], args.source_path or train_args["source_path"]
    )
    fourier = FourierOffset(gaussians.get_xyz.shape[0], ckpt["num_freqs"]).cuda()
    fourier.load_state_dict(ckpt["fourier"])
    sh = SHOffset(gaussians.get_xyz.shape[0], ckpt["num_sh_coeffs"], ckpt["num_freqs"]).cuda()
    sh.load_state_dict(ckpt["sh"])

    cams = make_camera_sequences(source_path, train_args["dataset"], 1, train_args["num_frames"], randomize=False)[0]
    distortions = list(ALL_DISTORTIONS) if args.distortions == ["all"] else (args.distortions or [])

    results, frames, gts = evaluate_watermark(
        cams, gaussians, pipe, background, decoder, message, fourier, sh, ImageMetrics(),
        clip_len=train_args["clip_len"], distortions=distortions, model_level=args.model_level, return_frames=True,
    )
    print(format_results(results))

    out_dir = args.out_dir or os.path.dirname(args.offsets)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if args.save_frames:
        fdir = os.path.join(out_dir, "frames")
        os.makedirs(fdir, exist_ok=True)
        to8 = lambda x: (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
        diffs = ((frames - gts).abs() * 10).clamp(0, 1)
        for tag, imgs in (("mark4d", to8(frames)), ("original", to8(gts)), ("diff_x10", to8(diffs))):
            for i, img in enumerate(imgs):
                Image.fromarray(img).save(os.path.join(fdir, f"{i:03d}_{tag}.png"), compress_level=1)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mark4D evaluation")
    p.add_argument("--offsets", required=True, help="offsets.pth written by train_watermark.py")
    p.add_argument("--decoder", required=True)
    p.add_argument("--source_path", default=None)
    p.add_argument("--distortions", nargs="*", default=None, help="names from mark4d/distortions.py or 'all'")
    p.add_argument("--model_level", action="store_true", help="also evaluate under model-level distortion")
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
