#
# depth_check.py
#
# Load a single depth image by path and plot its value distribution.
#
# Usage:
#   python depth_check.py --depth_path data/endonerf/cutting/depth/frame-000000.depth.png
#

import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # headless-safe: no X display needed
import matplotlib.pyplot as plt


def load_depth(depth_path):
    depth = np.array(Image.open(depth_path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def main():
    parser = ArgumentParser(description="Plot the depth distribution of a single depth image")
    parser.add_argument("--depth_path", type=str, required=True, help="path to a depth image")
    parser.add_argument("--out_dir", type=str, default="test", help="output directory for the plot")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--include_zero", action="store_true", help="include zero-depth (invalid) pixels")
    args = parser.parse_args()

    depth = load_depth(args.depth_path)
    valid = np.ones_like(depth, dtype=bool) if args.include_zero else depth != 0
    d = depth[valid]

    print(f"Depth image: {args.depth_path}")
    print(f"shape={depth.shape} dtype={depth.dtype}")
    print(f"valid pixels: {d.size}/{depth.size} ({100 * d.size / depth.size:.1f}%)")
    if d.size:
        print(f"min={d.min():.4f} p1={np.percentile(d, 1):.4f} median={np.median(d):.4f} "
              f"mean={d.mean():.4f} p99={np.percentile(d, 99):.4f} max={d.max():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    depth_vis = np.where(valid, depth, np.nan)
    im = axes[0].imshow(depth_vis, cmap="turbo")
    axes[0].set_title(os.path.basename(args.depth_path))
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(d.ravel(), bins=args.bins, color="steelblue")
    if d.size:
        axes[1].axvline(np.median(d), color="red", linestyle="--", linewidth=1, label=f"median={np.median(d):.2f}")
        axes[1].legend()
    axes[1].set_xlabel("Depth")
    axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Depth distribution")

    fig.tight_layout()
    out_name = os.path.splitext(os.path.basename(args.depth_path))[0] + "_depth_check.png"
    out_path = os.path.join(args.out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
