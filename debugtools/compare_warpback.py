#
# compare_warpback.py
#
# Bar-plot PSNR and SSIM per elevation for two experiments (grouped bars: one per experiment
# per elev), averaged over the offset views -- left/right/up/down by default; central is
# excluded since it's pose-invariant, not an offset view. Bars/error bars are computed by
# seaborn directly from the underlying per-frame samples (not pre-averaged). A paired t-test
# (matched by view + frame_idx, since both experiments render the same frames/views against
# the same ground truth) is run per elevation and annotated on the plot.
#
# Reads the per-frame warpback_gt_compare_elev<E>.json files (not the already-averaged
# warpback_summary.json) so the t-test has real per-frame samples to work with.
#
# Usage:
#   python debugtools/compare_warpback.py --exp1 endonerf/cutting --exp2 endonerf/cutting-depthreg
#

import os
import sys
import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

import matplotlib
matplotlib.use("Agg")  # headless-safe: no X display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root, for `from warpback import ...` regardless of cwd
from warpback import resolve_iteration


def load_per_frame(expname, elev, views, iteration=None):
    # {(view, frame_idx): {"psnr": ..., "ssim": ...}} for one elev, restricted to `views`
    sideview_root = os.path.join("output", expname, "sideview")
    it = resolve_iteration(sideview_root, iteration)
    path = os.path.join(sideview_root, f"ours_{it}", f"elev{elev:.0f}", f"warpback_gt_compare_elev{elev:.0f}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        results = json.load(f)
    return {(v["view"], v["frame_idx"]): v for v in results.values() if v["view"] in views}


def paired_values(rows1, rows2, metric):
    # only keys present (with a non-None metric) in BOTH experiments -- a real paired sample
    common = sorted(set(rows1) & set(rows2))
    pairs = [(rows1[k][metric], rows2[k][metric]) for k in common]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return np.array([]), np.array([])
    a, b = zip(*pairs)
    return np.array(a), np.array(b)


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    parser = ArgumentParser(description="Bar-plot warpback PSNR/SSIM per elevation for two experiments, with a paired t-test per elev")
    parser.add_argument("--exp1", required=True, type=str, help="e.g. endonerf/cutting (baseline)")
    parser.add_argument("--exp2", required=True, type=str, help="e.g. endonerf/cutting-depthreg (candidate)")
    parser.add_argument("--elev", nargs="+", type=float, default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--views", nargs="+", default=["left", "right", "up", "down"], help="views to average over (excludes central by default)")
    parser.add_argument("--iteration1", type=int, default=None)
    parser.add_argument("--iteration2", type=int, default=None)
    parser.add_argument("--out_dir", default="test")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric in zip(axes, ["psnr", "ssim"]):
        rows = []
        print(f"\n{metric}")
        print(f"{'elev':6s} {'n_pairs':8s} {args.exp1 + ' mean':>16s} {args.exp2 + ' mean':>16s} {'t-stat':>8s} {'p-value':>10s} {'sig':>4s}")
        annotations = {}
        for elev in args.elev:
            r1 = load_per_frame(args.exp1, elev, args.views, args.iteration1)
            r2 = load_per_frame(args.exp2, elev, args.views, args.iteration2)
            for k, v in r1.items():
                if v.get(metric) is not None:
                    rows.append({"elev": elev, "value": v[metric], "experiment": args.exp1})
            for k, v in r2.items():
                if v.get(metric) is not None:
                    rows.append({"elev": elev, "value": v[metric], "experiment": args.exp2})

            a, b = paired_values(r1, r2, metric)
            if len(a) >= 2:
                t_stat, p_val = stats.ttest_rel(a, b)
                print(f"{elev:<6.0f} {len(a):<8d} {a.mean():16.4f} {b.mean():16.4f} {t_stat:8.3f} {p_val:10.4g} {significance_stars(p_val):>4s}")
                annotations[elev] = significance_stars(p_val)
            else:
                print(f"{elev:<6.0f} {len(a):<8d} {'--':>16s} {'--':>16s} {'--':>8s} {'--':>10s} {'--':>4s}")

        df = pd.DataFrame(rows)
        sns.barplot(data=df, x="elev", y="value", hue="experiment", errorbar="sd", ax=ax)
        ax.set_xlabel("Elevation (deg)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.12 * (top - bottom))  # headroom for the significance annotations

        # annotate each elev group with the paired t-test significance
        elev_order = sorted(df["elev"].unique())
        for i, elev in enumerate(elev_order):
            if elev in annotations:
                group_max = df.loc[df["elev"] == elev, "value"].max()
                ax.text(i, group_max * 1.02, annotations[elev], ha="center", fontsize=10)

    fig.suptitle(
        f"{args.exp1} vs {args.exp2}, averaged over {', '.join(args.views)}\n"
        "paired t-test per elev: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(args.out_dir, exist_ok=True)
    safe = lambda s: s.replace("/", "_")
    out_name = f"warpback_compare_{safe(args.exp1)}_vs_{safe(args.exp2)}.png"
    out_path = os.path.join(args.out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
