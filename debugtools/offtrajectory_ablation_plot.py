#
# offtrajectory_ablation_plot.py
#
# For each sequence, plot PSNR (left) and SSIM (right) across elevations [0, 5, 10, 15, 20,
# 30, 45] for the depth x anisotropy ablation grid (GCSR+Aniso / GCSR only / Aniso only /
# No regularization), one line per configuration with a seaborn uncertainty band.
#
# elev=0 uses evaluate.py's per_view.json (the real, un-warped test-view render vs GT --
# same "orig" source compare_experiments.py uses). Every other elevation averages the 4
# offset views (left/right/up/down) together per frame first, same convention as
# compare_experiments.py's default, so the shaded band and this plot stay consistent with
# the paired t-tests already reported for this ablation grid.
#
# Usage:
#   python debugtools/offtrajectory_ablation_plot.py
#   python debugtools/offtrajectory_ablation_plot.py --sequences cutting
#

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling import
from compare_experiments import load_per_frame, load_original_view, average_per_frame

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # headless-safe: no X display needed
import matplotlib.pyplot as plt
from argparse import ArgumentParser

CONFIGS = {
    "cutting": [
        ("GCSR + Aniso", "endonerf/cutting-depthreg-aniso1e5-depth002"),
        ("GCSR only", "endonerf/cutting-depth002-noaniso"),
        ("Aniso only", "endonerf/cutting-aniso1e5-nodepth"),
        ("No reg", "endonerf/cutting-noaniso-nodepth"),
    ],
    "pulling": [
        ("GCSR + Aniso", "endonerf/pulling-depthreg-aniso1e5-depth002"),
        ("GCSR only", "endonerf/pulling-depth002-noaniso"),
        ("Aniso only", "endonerf/pulling-aniso1e5-nodepth"),
        ("No reg", "endonerf/pulling-noaniso-nodepth"),
    ],
}

ELEVS = [0, 5, 10, 15, 20, 30, 45]
OFFSET_VIEWS = ["left", "right", "up", "down"]


def build_rows(expname, metric):
    rows = []
    for elev in ELEVS:
        raw = load_original_view(expname) if elev == 0 else load_per_frame(expname, elev, OFFSET_VIEWS)
        avg = average_per_frame(raw)
        for (_, frame_idx), v in avg.items():
            if v.get(metric) is not None:
                rows.append({"elev": elev, "value": v[metric], "frame_idx": frame_idx})
    return rows


FONT_SIZES = {"axes.labelsize": 15, "axes.titlesize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12, "legend.title_fontsize": 13}


def main():
    parser = ArgumentParser(description="Plot PSNR/SSIM vs elevation for the depth x anisotropy ablation grid, one figure per sequence")
    parser.add_argument("--sequences", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    parser.add_argument("--errorbar", default="ci", choices=["ci", "sd"], help="ci: 95%% bootstrap CI on the mean (matches the paired t-tests); sd: spread of the raw per-frame data")
    parser.add_argument("--out_dir", default=os.path.join("output", "endonerf"))
    args = parser.parse_args()

    plt.rcParams.update(FONT_SIZES)

    for seq in args.sequences:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))  # 2/3 of the original (13, 5), so the fixed-point-size fonts/lines read larger
        for ax, metric in zip(axes, ["psnr", "ssim"]):
            all_rows = []
            for label, expname in CONFIGS[seq]:
                for row in build_rows(expname, metric):
                    all_rows.append({**row, "Config": label})
            df = pd.DataFrame(all_rows)
            sns.lineplot(data=df, x="elev", y="value", hue="Config", marker="o", errorbar=args.errorbar, ax=ax, legend=(metric == "psnr"))
            ax.set_title(metric.upper())
            ax.set_xlabel("Elevation (deg)")
            ax.set_ylabel("")
            ax.set_xticks(ELEVS)
            ax.set_ylim(10, 40) if metric == "psnr" else ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)

        fig.text(0.01, 0.5, seq.capitalize(), fontsize=16, rotation="vertical", va="center", ha="left")
        fig.tight_layout(rect=[0.03, 0, 1, 1])
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, f"offtrajectory_ablation_{seq}.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
