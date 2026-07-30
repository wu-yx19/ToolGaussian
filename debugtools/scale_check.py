#
# scale_check.py
#
# Load a trained Gaussian model (point_cloud.ply) and plot:
#   1. the distribution of the per-Gaussian max/min scale ratio (anisotropy)
#   2. the distribution of the per-Gaussian max scale (size)
#
# Reads the raw "scale_*" ply properties directly, so no CUDA/torch is needed.
# Ply scales are stored in log-space (see GaussianModel.scaling_activation = exp),
# so we exponentiate them to get real-world scale magnitudes.
#
# By default, the log and plot are written under the model's own output
# folder (<model_path>/scale_check/<iteration_N>/), inferred from the
# standard <model_path>/point_cloud/<iteration_N>/point_cloud.ply layout,
# rather than a shared "test" folder.
#
# Usage:
#   python debugtools/scale_check.py --ply_path output/endonerf/cutting/point_cloud/iteration_3000/point_cloud.ply
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
    return os.path.join(model_path, "scale_check", os.path.basename(iteration_dir))


def load_scales(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]

    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    assert scale_names, f"no scale_* properties found in {ply_path}"

    scales_log = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1)
    return np.exp(scales_log)  # undo scaling_inverse_activation (log)


def main():
    parser = ArgumentParser(description="Plot max/min scale ratio and max scale distribution of a trained Gaussian model")
    parser.add_argument("--ply_path", type=str, required=True, help="path to a trained point_cloud.ply")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="output directory for the log/plot (default: <model_path>/scale_check/<iteration_N>/)")
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir else infer_out_dir(args.ply_path)
    os.makedirs(out_dir, exist_ok=True)

    scales = load_scales(args.ply_path)  # (N, 3), real-world scale magnitudes
    max_scale = scales.max(axis=1)
    min_scale = scales.min(axis=1)
    ratio = max_scale / min_scale

    lines = [
        f"Ply: {args.ply_path}",
        f"num gaussians: {scales.shape[0]}",
        f"max scale:  min={max_scale.min():.4g} median={np.median(max_scale):.4g} "
        f"mean={max_scale.mean():.4g} p99={np.percentile(max_scale, 99):.4g} max={max_scale.max():.4g}",
        f"max/min ratio: min={ratio.min():.4g} median={np.median(ratio):.4g} "
        f"mean={ratio.mean():.4g} p99={np.percentile(ratio, 99):.4g} max={ratio.max():.4g}",
    ]
    for line in lines:
        print(line)

    log_name = os.path.splitext(os.path.basename(args.ply_path))[0] + "_scale_check.log"
    log_path = os.path.join(out_dir, log_name)
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved log to {log_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    axes[1].set_xlabel("max scale (log10)")
    axes[1].set_ylabel("Gaussian count")
    axes[1].set_title("Max scale distribution")
    axes[1].legend()

    fig.suptitle(os.path.basename(args.ply_path))
    fig.tight_layout()
    out_name = os.path.splitext(os.path.basename(args.ply_path))[0] + "_scale_check.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
