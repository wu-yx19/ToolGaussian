#
# compare_experiments.py
#
# Bar-plot PSNR and SSIM per elevation for two experiments (grouped bars: one per experiment
# per elev), averaged over the offset views -- left/right/up/down by default; central is
# excluded since it's pose-invariant, not an offset view. A leading "orig" group adds the
# real, un-warped test-view result (evaluate.py's per_view.json) alongside the sideview elevs.
#
# By default, each frame's views are averaged together BEFORE the t-test/plot (one sample per
# frame, e.g. 20 samples/elev), since a frame's 4 offset views share the same underlying
# reconstruction at that time-step and aren't independent -- pooling all (view, frame) pairs
# directly (e.g. 80 samples/elev) pseudo-replicates and overstates significance. Pass
# --pool_views to use the pooled samples instead (more power, less statistically defensible).
#
# A paired t-test (matched by frame, since both experiments render the same frames against the
# same ground truth) is run per group and annotated on the plot.
#
# Reads the per-frame warpback_gt_compare_elev<E>.json files score_against_gt.py writes (not
# the already-averaged warpback_summary.json) so the t-test has real per-frame samples to work with.
#
# Usage:
#   python debugtools/compare_experiments.py --exp1 endonerf/cutting --exp2 endonerf/cutting-depthreg
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root, for `from warp_to_source import ...` regardless of cwd
from warp_to_source import resolve_iteration


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


def load_original_view(expname, iteration=None):
    # {("original", frame_idx): {"view": "original", "frame_idx": ..., "psnr": ..., "ssim": ...}}
    # sourced from evaluate.py's per_view.json (the real, un-warped test-view render vs GT);
    # PSNR* is the tool-masked variant (matches the masked convention used elsewhere here);
    # there's no masked SSIM variant in per_view.json, so SSIM here is unmasked -- not a
    # perfectly apples-to-apples comparison against the sideview SSIM, worth noting when reporting.
    # Filenames are true frame indices (render.py/write_images names test-set output by
    # view.uid, not by list position) -- requires a render.py + evaluate.py rerun for any
    # experiment whose per_view.json predates that fix.
    path = os.path.join("output", expname, "per_view.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    it = iteration if iteration is not None else int(resolve_iteration(os.path.join("output", expname, "sideview"), None))
    key = f"ours_{it}"
    if key not in d:
        key = next(iter(d))  # fall back to whatever single method dir evaluate.py found
    metrics = d[key]
    out = {}
    for fname, psnr_val in metrics["PSNR*"].items():
        frame_idx = int(os.path.splitext(fname)[0])
        out[("original", frame_idx)] = {
            "view": "original", "frame_idx": frame_idx,
            "psnr": psnr_val, "ssim": metrics["SSIM"].get(fname),
        }
    return out


def average_per_frame(rows):
    # {(view, frame_idx): {...}} -> {("avg", frame_idx): {...}}, averaging psnr/ssim across
    # whatever views are present for that frame. The 4 offset views of one frame share the
    # same underlying reconstruction at that time-step (a hard frame drags all 4 down together),
    # so treating them as independent samples in a t-test pseudo-replicates and understates
    # p-values; averaging first makes "frame" the unit of independence. A no-op for groups that
    # already have exactly one view per frame (e.g. "original"), since mean-of-one = itself.
    by_frame = {}
    for (view, frame_idx), v in rows.items():
        bucket = by_frame.setdefault(frame_idx, {"psnr": [], "ssim": []})
        if v.get("psnr") is not None:
            bucket["psnr"].append(v["psnr"])
        if v.get("ssim") is not None:
            bucket["ssim"].append(v["ssim"])
    return {
        ("avg", frame_idx): {
            "view": "avg", "frame_idx": frame_idx,
            "psnr": float(np.mean(vals["psnr"])) if vals["psnr"] else None,
            "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else None,
        }
        for frame_idx, vals in by_frame.items()
    }


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


FONT_SIZES = {"axes.labelsize": 15, "axes.titlesize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12, "legend.title_fontsize": 13}


def main():
    parser = ArgumentParser(description="Bar-plot RTVC (warp-to-source) PSNR/SSIM per elevation for two experiments, with a paired t-test per elev -- same visual style as offtrajectory_ablation_plot.py")
    parser.add_argument("--exp1", required=True, type=str, help="e.g. endonerf/cutting-noaniso-nodepth (baseline)")
    parser.add_argument("--exp2", required=True, type=str, help="e.g. endonerf/cutting-depthreg-aniso1e5-depth002 (candidate)")
    parser.add_argument("--exp1_label", default="Baseline")
    parser.add_argument("--exp2_label", default="OffTrackGS (Ours)")
    parser.add_argument("--elev", nargs="+", type=float, default=[5, 10, 15, 20, 30, 45])
    parser.add_argument("--views", nargs="+", default=["left", "right", "up", "down"], help="views to average over (excludes central by default)")
    parser.add_argument("--iteration1", type=int, default=None)
    parser.add_argument("--iteration2", type=int, default=None)
    parser.add_argument("--pool_views", action="store_true", help="t-test/plot over all (view, frame) pairs directly (e.g. 4x20=80 samples/elev) instead of averaging the views per frame first (20 samples/elev, default) -- pooling pseudo-replicates since a frame's 4 offset views aren't independent, so it's more powerful but overstates significance")
    parser.add_argument("--seq_label", default=None, help="optional sequence name shown vertically on the left, e.g. Cutting")
    parser.add_argument("--out_dir", default=None, help="default: output/<exp2> (the candidate experiment's own folder)")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("output", args.exp2)

    plt.rcParams.update(FONT_SIZES)

    # "orig" (the real, un-warped test-view render vs GT, from evaluate.py's per_view.json)
    # goes first, alongside the sideview elevations. Its n is ~4x smaller than each sideview
    # elev's, since sideview aggregates 4 offset views x N frames while original is just N frames.
    groups = [("0", None)] + [(f"{e:.0f}", e) for e in args.elev]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, metric in zip(axes, ["psnr", "ssim"]):
        rows = []
        print(f"\n{metric}")
        print(f"{'group':6s} {'n_pairs':8s} {args.exp1_label + ' mean':>16s} {args.exp2_label + ' mean':>16s} {'t-stat':>8s} {'p-value':>10s} {'sig':>4s}")
        annotations = {}
        for label, elev in groups:
            if elev is None:
                r1 = load_original_view(args.exp1, args.iteration1)
                r2 = load_original_view(args.exp2, args.iteration2)
            else:
                r1 = load_per_frame(args.exp1, elev, args.views, args.iteration1)
                r2 = load_per_frame(args.exp2, elev, args.views, args.iteration2)
            if not args.pool_views:
                r1 = average_per_frame(r1)
                r2 = average_per_frame(r2)
            for k, v in r1.items():
                if v.get(metric) is not None:
                    rows.append({"group": label, "value": v[metric], "Config": args.exp1_label})
            for k, v in r2.items():
                if v.get(metric) is not None:
                    rows.append({"group": label, "value": v[metric], "Config": args.exp2_label})

            a, b = paired_values(r1, r2, metric)
            if len(a) >= 2:
                t_stat, p_val = stats.ttest_rel(a, b)
                print(f"{label:<6s} {len(a):<8d} {a.mean():16.4f} {b.mean():16.4f} {t_stat:8.3f} {p_val:10.4g} {significance_stars(p_val):>4s}")
                annotations[label] = significance_stars(p_val)
            else:
                print(f"{label:<6s} {len(a):<8d} {'--':>16s} {'--':>16s} {'--':>8s} {'--':>10s} {'--':>4s}")

        df = pd.DataFrame(rows)
        group_order = [label for label, _ in groups if label in set(df["group"])]
        palette = {args.exp2_label: "tab:blue", args.exp1_label: "tab:red"}
        sns.barplot(data=df, x="group", y="value", hue="Config", hue_order=[args.exp2_label, args.exp1_label], palette=palette, order=group_order, errorbar="sd", ax=ax, legend=(metric == "psnr"))
        ax.set_title(metric.upper())
        ax.set_xlabel("Elevation (deg)")
        ax.set_ylabel("")
        ax.set_ylim(10, 40) if metric == "psnr" else ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.08 * (top - bottom))  # headroom for the significance annotations

        # annotate each group with the paired t-test significance
        for i, label in enumerate(group_order):
            if label in annotations:
                group_max = df.loc[df["group"] == label, "value"].max()
                ax.text(i, group_max + 0.02 * (top - bottom), annotations[label], ha="center", fontsize=10)

    if args.seq_label:
        fig.text(0.01, 0.5, args.seq_label, fontsize=16, rotation="vertical", va="center", ha="left")
    fig.tight_layout(rect=[0.03, 0, 1, 1] if args.seq_label else [0, 0, 1, 1])
    os.makedirs(args.out_dir, exist_ok=True)
    safe = lambda s: s.replace("/", "_")
    out_name = f"compare_experiments_{safe(args.exp1)}_vs_{safe(args.exp2)}.png"
    out_path = os.path.join(args.out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
