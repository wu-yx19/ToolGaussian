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


def gradient_stats(name, g):
    # summary stats on |gradient|, printed to help pick a huber_loss beta: beta should sit
    # somewhere between the "typical" magnitude (median/p90, real smooth-surface noise -- want
    # these squashed quadratically) and the "outlier" magnitude (p99/max, real discontinuities
    # like tissue folds or floater edges -- want these only penalized linearly, not exploded)
    ag = np.abs(g)
    print(f"|{name}|: min={ag.min():.4f} median={np.median(ag):.4f} mean={ag.mean():.4f} "
          f"p90={np.percentile(ag, 90):.4f} p99={np.percentile(ag, 99):.4f} max={ag.max():.4f}")


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

    # dh/dw: same neighbor-difference huber_loss (utils/loss_utils.py) computes on rendered
    # depth; only keep a difference where both pixels it's computed from are valid, so a jump
    # across a real invalid-depth hole doesn't masquerade as a genuine discontinuity
    dh = depth[1:, :] - depth[:-1, :]
    dh_valid = (valid[1:, :] & valid[:-1, :])
    dh = dh[dh_valid]
    dw = depth[:, 1:] - depth[:, :-1]
    dw_valid = (valid[:, 1:] & valid[:, :-1])
    dw = dw[dw_valid]

    if dh.size:
        gradient_stats("dh", dh)
    if dw.size:
        gradient_stats("dw", dw)

    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    depth_vis = np.where(valid, depth, np.nan)
    im = axes[0, 0].imshow(depth_vis, cmap="turbo")
    axes[0, 0].set_title(os.path.basename(args.depth_path))
    axes[0, 0].axis("off")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[0, 1].hist(d.ravel(), bins=args.bins, color="steelblue")
    if d.size:
        axes[0, 1].axvline(np.median(d), color="red", linestyle="--", linewidth=1, label=f"median={np.median(d):.2f}")
        axes[0, 1].legend()
    axes[0, 1].set_xlabel("Depth")
    axes[0, 1].set_ylabel("Pixel count")
    axes[0, 1].set_title("Depth distribution")

    for ax, (name, g) in zip(axes[1], [("dh", dh), ("dw", dw)]):
        if g.size:
            ax.hist(g.ravel(), bins=args.bins, color="darkorange")
            median, p90, p99 = np.median(np.abs(g)), np.percentile(np.abs(g), 90), np.percentile(np.abs(g), 99)
            for val, color, label in [
                (median, "red", f"median|.|={median:.3f}"),
                (p90, "green", f"p90|.|={p90:.3f}"),
                (p99, "purple", f"p99|.|={p99:.3f}"),
            ]:
                ax.axvline(val, color=color, linestyle="--", linewidth=1, label=label)
                ax.axvline(-val, color=color, linestyle="--", linewidth=1)
            ax.legend(fontsize=8)
        ax.set_xlabel(f"{name} (neighbor depth difference)")
        ax.set_ylabel("Pixel-pair count")
        ax.set_title(f"{name} distribution (candidate huber beta values marked)")

    fig.tight_layout()
    out_name = os.path.splitext(os.path.basename(args.depth_path))[0] + "_depth_check.png"
    out_path = os.path.join(args.out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
