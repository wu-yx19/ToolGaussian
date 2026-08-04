#
# scale_opacity_check.py
#
# Load a trained Gaussian model (point_cloud.ply) and plot:
#   1. the distribution of the per-Gaussian max/min scale ratio (anisotropy)
#   2. the distribution of the per-Gaussian max scale (size)
#   3. the distribution of the per-Gaussian opacity
#
# Reads the raw ply properties directly, so no CUDA/torch is needed.
# Ply scales are stored in log-space (GaussianModel.scaling_activation = exp)
# and opacity is stored in logit-space (GaussianModel.opacity_activation = sigmoid),
# so we invert both to get real-world scale magnitudes and [0,1] opacities.
#
# Also prints, for reference, what fraction of points would fall on either
# side of the current size/opacity thresholds used by scene.gaussians.prune()
# and reset_opacity() -- useful for picking reasonable threshold values instead
# of guessing.
#
# Note: this only covers the world-space size criterion (scale vs. extent).
# The other size criterion in prune(), max_radii2D > prune_size_threshold, is a
# screen-space pixel radius that depends on which camera views a Gaussian is
# visible from -- it isn't stored in the ply and would need an actual render
# pass over the camera set to measure, which this script doesn't do.
#
# By default, the log and plot are written under the model's own output
# folder (<model_path>/scale_opacity_check/<iteration_N>/), inferred from
# the standard <model_path>/point_cloud/<iteration_N>/point_cloud.ply layout,
# rather than a shared "test" folder.
#
# Usage:
#   python debugtools/scale_opacity_check.py \
#       --ply_path output/endonerf-org/cutting/point_cloud/iteration_3000/point_cloud.ply
#

import os
from argparse import ArgumentParser

import numpy as np
from plyfile import PlyData

import matplotlib
matplotlib.use("Agg")  # headless-safe: no X display needed
import matplotlib.pyplot as plt


def infer_out_dir(ply_path):
    iteration_dir = os.path.dirname(ply_path)      # .../point_cloud/iteration_N
    point_cloud_dir = os.path.dirname(iteration_dir)  # .../point_cloud
    model_path = os.path.dirname(point_cloud_dir)     # model output folder
    return os.path.join(model_path, "scale_opacity_check", os.path.basename(iteration_dir))


def load_gaussian_properties(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]

    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    assert scale_names, f"no scale_* properties found in {ply_path}"

    scales_log = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1)
    scales = np.exp(scales_log)  # undo scaling_inverse_activation (log)

    opacity_logit = np.asarray(vertex["opacity"])
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))  # undo inverse_sigmoid

    return scales, opacity


def main():
    parser = ArgumentParser(description="Plot scale (anisotropy/size) and opacity distributions of a trained Gaussian model")
    parser.add_argument("--ply_path", type=str, required=True, help="path to a trained point_cloud.ply")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="output directory for the log/plot (default: <model_path>/scale_opacity_check/<iteration_N>/)")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--extent", type=float, default=10.0, help="scene.cameras_extent, for the world-space size threshold reference line")
    parser.add_argument("--scale_extent_ratio", type=float, default=0.1, help="prune_scale_extent_ratio reference value (threshold = ratio * extent)")
    parser.add_argument("--opacity_threshold", type=float, default=0.005, help="opacity_threshold_* reference value (points below this get pruned)")
    parser.add_argument("--opacity_reset_value", type=float, default=0.01, help="opacity_reset_value reference value (ceiling applied by reset_opacity)")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir else infer_out_dir(args.ply_path)
    os.makedirs(out_dir, exist_ok=True)

    scales, opacity = load_gaussian_properties(args.ply_path)  # scales: (N, 3) real-world magnitudes, opacity: (N,) in [0,1]
    max_scale = scales.max(axis=1)
    min_scale = scales.min(axis=1)
    ratio = max_scale / min_scale

    size_threshold = args.scale_extent_ratio * args.extent
    pct_above_size_threshold = 100.0 * np.mean(max_scale > size_threshold)
    pct_below_opacity_threshold = 100.0 * np.mean(opacity < args.opacity_threshold)
    pct_below_opacity_reset_value = 100.0 * np.mean(opacity < args.opacity_reset_value)

    lines = [
        f"Ply: {args.ply_path}",
        f"num gaussians: {scales.shape[0]}",
        f"max scale (size): min={max_scale.min():.4g} median={np.median(max_scale):.4g} "
        f"mean={max_scale.mean():.4g} p99={np.percentile(max_scale, 99):.4g} max={max_scale.max():.4g}",
        f"max/min ratio (anisotropy): min={ratio.min():.4g} median={np.median(ratio):.4g} "
        f"mean={ratio.mean():.4g} p99={np.percentile(ratio, 99):.4g} max={ratio.max():.4g}",
        f"opacity: min={opacity.min():.4g} median={np.median(opacity):.4g} "
        f"mean={opacity.mean():.4g} p1={np.percentile(opacity, 1):.4g} max={opacity.max():.4g}",
        f"world-space size threshold = scale_extent_ratio({args.scale_extent_ratio}) * extent({args.extent}) = {size_threshold:.4g} "
        f"-> {pct_above_size_threshold:.2f}% of points would be pruned by this criterion alone",
        f"opacity_threshold={args.opacity_threshold} -> {pct_below_opacity_threshold:.2f}% of points already below it",
        f"opacity_reset_value={args.opacity_reset_value} -> {pct_below_opacity_reset_value:.2f}% of points already below it "
        f"(i.e. would be unaffected by a reset_opacity() ceiling at this value)",
    ]
    for line in lines:
        print(line)

    log_name = os.path.splitext(os.path.basename(args.ply_path))[0] + "_scale_opacity_check.log"
    log_path = os.path.join(out_dir, log_name)
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved log to {log_path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ratio spans several orders of magnitude (long tail of needle-like gaussians),
    # so a linear axis crams almost everything into the first bin; use log10 instead.
    log_ratio = np.log10(ratio)
    axes[0].hist(log_ratio, bins=args.bins, color="steelblue")
    axes[0].axvline(np.median(log_ratio), color="red", linestyle="--", linewidth=1,
                     label=f"median={np.median(ratio):.2f}")
    axes[0].set_xlabel("max/min scale ratio (log10)")
    axes[0].set_ylabel("Gaussian count")
    axes[0].set_title("Anisotropy (max/min scale ratio)")
    axes[0].legend()

    log_max_scale = np.log10(max_scale)
    axes[1].hist(log_max_scale, bins=args.bins, color="darkorange")
    axes[1].axvline(np.median(log_max_scale), color="red", linestyle="--", linewidth=1,
                     label=f"median={np.median(max_scale):.3g}")
    axes[1].axvline(np.log10(size_threshold), color="black", linestyle=":", linewidth=1.5,
                     label=f"size threshold={size_threshold:.3g} ({pct_above_size_threshold:.2f}% above)")
    axes[1].set_xlabel("max scale / size (log10)")
    axes[1].set_ylabel("Gaussian count")
    axes[1].set_title("Size distribution (world-space scale)")
    axes[1].legend()

    axes[2].hist(opacity, bins=args.bins, color="seagreen", range=(0, 1))
    axes[2].axvline(np.median(opacity), color="red", linestyle="--", linewidth=1,
                     label=f"median={np.median(opacity):.3g}")
    axes[2].axvline(args.opacity_threshold, color="black", linestyle=":", linewidth=1.5,
                     label=f"opacity_threshold={args.opacity_threshold} ({pct_below_opacity_threshold:.2f}% below)")
    axes[2].axvline(args.opacity_reset_value, color="purple", linestyle=":", linewidth=1.5,
                     label=f"opacity_reset_value={args.opacity_reset_value} ({pct_below_opacity_reset_value:.2f}% below)")
    axes[2].set_xlabel("opacity")
    axes[2].set_ylabel("Gaussian count")
    axes[2].set_title("Opacity distribution")
    axes[2].legend()

    fig.suptitle(os.path.basename(args.ply_path))
    fig.tight_layout()
    out_name = os.path.splitext(os.path.basename(args.ply_path))[0] + "_scale_opacity_check.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
