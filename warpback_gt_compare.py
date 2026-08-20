#
# warpback_gt_compare.py
#
# Compare warpback.py's outputs (a render warped back to its original camera)
# against the frame's actual ground-truth image, restricted to pixels valid for
# BOTH: the warp's own coverage mask (warpback_*_mask.png) and the tool mask --
# GT under the tool is unreliable/excluded throughout this codebase (see
# psnr(..., mask) in utils/eval_utils.py).
#
# Both the GT image and the tool mask are read from render.py's own output
# (output/<expname>/video/ours_<iteration>/{gt,masks}/<frame_idx>.png), i.e.
# view.original_image / view.mask as actually used by the rest of this
# codebase's training/eval code (mask: white=valid), rather than the raw
# dataset files directly. This sidesteps two dataset-specific gotchas: image
# naming varies by clip (pulling: frame-000000.color.png, cutting: 000000.png)
# but render.py's output is always <frame_idx:05d>.png regardless of clip; and
# the raw mask file has the OPPOSITE polarity (white=tool) from view.mask
# (white=valid) used everywhere else. Requires having run render.py (which
# renders train/test/video by default) for this expname/iteration at least once.
#
# Usage:
#   python warpback_gt_compare.py --expname endonerf/pulling --elev 20
#

import os
import re
import glob
import json
import argparse

import numpy as np
import cv2

from utils.image_utils import sideview_view_elevs, masked_ssim_map, masked_psnr
from warpback import resolve_iteration


def default_mask_path(warpback_path):
    root, ext = os.path.splitext(warpback_path)
    return f"{root}_mask{ext}"


def find_warpback_files(warpback_dir, view_name):
    # warpback_frame<F>_<view_name>_azim..._elev..._dist....png -> (frame_idx, path)
    pattern = os.path.join(warpback_dir, f"warpback_frame*_{view_name}_*.png")
    results = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith("_mask.png"):
            continue
        m = re.match(r"warpback_frame(\d+)_", os.path.basename(path))
        if m:
            results.append((int(m.group(1)), path))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare warpback.py outputs against ground truth, over pixels valid for both the warp and the tool mask")
    parser.add_argument("--expname", required=True, type=str, help="e.g. endonerf/pulling -> output/endonerf/pulling")
    parser.add_argument("--iteration", type=int, default=None, help="the ours_<iteration> checkpoint (auto-detected if there's only one; must have been rendered by both sideview.py and render.py)")
    parser.add_argument("--elev", required=True, type=float, nargs="+", help="one or more elevations to process")
    parser.add_argument("--ssim_window", type=int, default=11, help="SSIM window size in pixels")
    parser.add_argument("--coverage_thresh", type=float, default=0.5, help="min fraction of an SSIM window that must be valid to trust it")
    args = parser.parse_args()

    sideview_root = os.path.join("output", args.expname, "sideview")
    iteration = resolve_iteration(sideview_root, args.iteration)
    sideview_dir = os.path.join(sideview_root, f"ours_{iteration}")

    video_dir = os.path.join("output", args.expname, "video", f"ours_{iteration}")
    gt_dir = os.path.join(video_dir, "gt")
    tool_mask_dir = os.path.join(video_dir, "masks")
    if not os.path.isdir(gt_dir) or not os.path.isdir(tool_mask_dir):
        raise FileNotFoundError(f"missing {gt_dir} or {tool_mask_dir} -- run render.py --expname {args.expname} first (it renders train/test/video, including these)")

    any_results = False
    summary = {}  # {view_name: {elev: {"psnr": [...], "ssim": [...]}}}, collapsed to means at the end
    for elev in args.elev:
        # results for this elev (including "central", sourced from elev0/) land under this
        # elev's own folder -- same convention as sideview.py's own concat placement
        results = {}
        for view_name, view_elev in sideview_view_elevs(elev):
            view_elev_dir = os.path.join(sideview_dir, f"elev{view_elev:.0f}")
            warpback_dir = os.path.join(view_elev_dir, "warpback")
            if not os.path.isdir(warpback_dir):
                continue

            for frame_idx, warpback_path in find_warpback_files(warpback_dir, view_name):
                warp_mask_path = default_mask_path(warpback_path)
                if not os.path.isfile(warp_mask_path):
                    print(f"skipping {warpback_path}: no warpback mask found")
                    continue

                gt_path = os.path.join(gt_dir, f"{frame_idx:05d}.png")
                tool_mask_path = os.path.join(tool_mask_dir, f"{frame_idx:05d}.png")
                if not (os.path.isfile(gt_path) and os.path.isfile(tool_mask_path)):
                    print(f"skipping {warpback_path}: no GT/mask found ({gt_path})")
                    continue

                warpback_img = cv2.imread(warpback_path)
                gt_img = cv2.imread(gt_path)
                if warpback_img.shape != gt_img.shape:
                    raise ValueError(f"{warpback_path}: shape {warpback_img.shape} != GT shape {gt_img.shape} ({gt_path})")

                warp_valid = cv2.imread(warp_mask_path, cv2.IMREAD_GRAYSCALE) > 127
                # render.py's saved view.mask: white(255)=valid tissue, black(0)=tool
                tool_valid = cv2.imread(tool_mask_path, cv2.IMREAD_GRAYSCALE) > 127
                valid = warp_valid & tool_valid

                if not valid.any():
                    print(f"skipping {warpback_path}: no pixels valid for both warp and GT")
                    continue

                warpback_gray = cv2.cvtColor(warpback_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

                ssim_map, coverage = masked_ssim_map(warpback_gray, gt_gray, valid.astype(np.float32), args.ssim_window)
                valid_ssim = valid & (coverage >= args.coverage_thresh)

                psnr_score = masked_psnr(warpback_img, gt_img, valid)
                ssim_score = float(ssim_map[valid_ssim].mean()) if valid_ssim.any() else None

                key = f"{view_name}_frame{frame_idx}_elev{elev:.0f}"
                results[key] = {
                    "view": view_name, "frame_idx": frame_idx, "elev": elev,
                    "valid_pixel_fraction": float(valid.mean()),
                    "psnr": psnr_score, "ssim": ssim_score,
                }
                print(f"{key}: psnr={psnr_score:.2f} ssim={ssim_score:.4f} valid={valid.mean():.1%}" if ssim_score is not None
                      else f"{key}: psnr={psnr_score:.2f} ssim=n/a (no valid SSIM windows) valid={valid.mean():.1%}")

                bucket = summary.setdefault(view_name, {}).setdefault(elev, {"psnr": [], "ssim": []})
                bucket["psnr"].append(psnr_score)
                if ssim_score is not None:
                    bucket["ssim"].append(ssim_score)

        if not results:
            continue
        any_results = True

        out_dir = os.path.join(sideview_dir, f"elev{elev:.0f}")
        out_path = os.path.join(out_dir, f"warpback_gt_compare_elev{elev:.0f}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out_path}")

    if not any_results:
        raise FileNotFoundError("no warpback+GT pairs found to compare")

    # one summary file across every requested elev, averaged over frames per (view, elev),
    # in the experiment root next to results.json (see evaluate.py's own scene_dir placement)
    summary_out = {
        view_name: {
            f"elev{elev:.0f}": {
                "psnr_mean": float(np.mean(bucket["psnr"])),
                "ssim_mean": float(np.mean(bucket["ssim"])) if bucket["ssim"] else None,
                "n_frames": len(bucket["psnr"]),
            }
            for elev, bucket in sorted(elev_buckets.items())
        }
        for view_name, elev_buckets in summary.items()
    }
    # merge into any existing summary rather than overwriting it -- a later run requesting a
    # different --elev (e.g. 5 15 25 after an earlier 10 20 30) should add to it, not lose the
    # elevs it didn't touch this time; per (view, elev) still overwrites if re-run on purpose
    summary_path = os.path.join("output", args.expname, "warpback_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
        for view_name, elev_dict in summary_out.items():
            existing.setdefault(view_name, {}).update(elev_dict)
        summary_out = existing
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
