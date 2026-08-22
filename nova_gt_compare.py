#
# nova_gt_compare.py
#
# Compare sideview.py's renders directly against ground truth using NOVA's
# non-aligned reference IQA (cosine distance between fine-tuned DINOv2
# embeddings) instead of pixel-aligned PSNR/SSIM.
#
# Unlike score_against_gt.py, no warp_to_source.py alignment step is required:
# a sideview render sits at an offset camera pose (different from the frame's
# GT pose), which is exactly the non-aligned-reference case NOVA is built
# for, so it's compared straight against GT.
#
# GT is read from render.py's own output (output/<expname>/video/ours_<iteration>/gt/
# <frame_idx:05d>.png), same source score_against_gt.py uses, so frame_idx
# numbering matches sideview.py's video-view indexing regardless of clip.
#
# This only needs torch/timm/PIL/numpy (NOVA's requirements.txt) -- it
# deliberately avoids importing utils.image_utils / warp_to_source.py, since those
# pull in cv2/imageio which aren't installed in the NOVA container
# (docker/nova_env.sif). The couple of helpers duplicated from warp_to_source.py
# below are intentionally kept in sync with it rather than imported.
#
# Usage (inside the NOVA apptainer container, see sbatch/nova_gt_compare_single.sh):
#   python nova_gt_compare.py --expname endonerf/cutting --elev 10 20 30
#

import os
import re
import glob
import json
import sys
import argparse

import numpy as np

NOVA_REPO = os.environ.get("NOVA_REPO", "/home/groups/bdaniel/wyx/Projects/NOVA")
sys.path.insert(0, NOVA_REPO)
from nova import load_model, compute_cosine_distance, pick_device  # noqa: E402


def sideview_view_elevs(elev):
    # mirrors sideview.py's get_view_offsets / utils.image_utils.sideview_view_elevs:
    # "central" always renders (and lands) at elev0/ regardless of the requested elev
    return [("central", 0.0), ("left", elev), ("right", elev), ("up", elev), ("down", elev)]


def resolve_iteration(sideview_root, iteration):
    # mirrors warp_to_source.py's resolve_iteration
    if iteration is not None:
        return iteration
    candidates = sorted(int(m.group(1)) for d in glob.glob(os.path.join(sideview_root, "ours_*"))
                         if (m := re.fullmatch(r"ours_(\d+)", os.path.basename(d))))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no ours_<iteration> dirs found under {sideview_root}")
    raise ValueError(f"multiple ours_<iteration> dirs found under {sideview_root} ({candidates}); pass --iteration to pick one")


def find_sideview_renders(view_elev_dir, view_name):
    # render_frame<F>_<view_name>_azim..._elev..._dist....png -> (frame_idx, path)
    pattern = os.path.join(view_elev_dir, "renders", f"render_frame*_{view_name}_*.png")
    results = []
    for path in sorted(glob.glob(pattern)):
        m = re.match(r"render_frame(\d+)_", os.path.basename(path))
        if m:
            results.append((int(m.group(1)), path))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare sideview.py renders against ground truth via NOVA's non-aligned reference IQA")
    parser.add_argument("--expname", required=True, type=str, help="e.g. endonerf/pulling -> output/endonerf/pulling")
    parser.add_argument("--iteration", type=int, default=None, help="the ours_<iteration> checkpoint (auto-detected if there's only one; must have been rendered by both sideview.py and render.py)")
    parser.add_argument("--elev", required=True, type=float, nargs="+", help="one or more elevations to process")
    parser.add_argument("--nova-checkpoint", type=str, default=os.path.join(NOVA_REPO, "weights", "NOVA_NVS.pt"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    sideview_root = os.path.join("output", args.expname, "sideview")
    iteration = resolve_iteration(sideview_root, args.iteration)
    sideview_dir = os.path.join(sideview_root, f"ours_{iteration}")

    gt_dir = os.path.join("output", args.expname, "video", f"ours_{iteration}", "gt")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"missing {gt_dir} -- run render.py --expname {args.expname} first (it renders train/test/video, including GT)")

    device = pick_device(args.device)
    model = load_model(checkpoint_path=args.nova_checkpoint, device=device)

    any_results = False
    summary = {}  # {view_name: {elev: [cosine_distance, ...]}}
    for elev in args.elev:
        results = {}
        for view_name, view_elev in sideview_view_elevs(elev):
            view_elev_dir = os.path.join(sideview_dir, f"elev{view_elev:.0f}")
            renders = find_sideview_renders(view_elev_dir, view_name)
            if not renders:
                continue

            for frame_idx, render_path in renders:
                gt_path = os.path.join(gt_dir, f"{frame_idx:05d}.png")
                if not os.path.isfile(gt_path):
                    print(f"skipping {render_path}: no GT found ({gt_path})")
                    continue

                # image_a=GT is the (non-aligned) reference, image_b=sideview render is being evaluated
                score = compute_cosine_distance(model, gt_path, render_path, device)

                key = f"{view_name}_frame{frame_idx}_elev{elev:.0f}"
                results[key] = {
                    "view": view_name, "frame_idx": frame_idx, "elev": elev,
                    "cosine_distance": score["cosine_distance"],
                    "cosine_similarity": score["cosine_similarity"],
                }
                print(f"{key}: cosine_distance={score['cosine_distance']:.4f} cosine_similarity={score['cosine_similarity']:.4f}")

                bucket = summary.setdefault(view_name, {}).setdefault(elev, [])
                bucket.append(score["cosine_distance"])

        if not results:
            continue
        any_results = True

        out_dir = os.path.join(sideview_dir, f"elev{elev:.0f}")
        out_path = os.path.join(out_dir, f"nova_gt_compare_elev{elev:.0f}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out_path}")

    if not any_results:
        raise FileNotFoundError("no sideview+GT pairs found to compare")

    # one summary file across every requested elev, averaged over frames per (view, elev),
    # merged into any existing summary rather than overwriting it (same convention as
    # score_against_gt.py's warpback_summary.json)
    summary_out = {
        view_name: {
            f"elev{elev:.0f}": {
                "cosine_distance_mean": float(np.mean(dists)),
                "n_frames": len(dists),
            }
            for elev, dists in sorted(elev_buckets.items())
        }
        for view_name, elev_buckets in summary.items()
    }
    summary_path = os.path.join("output", args.expname, "nova_summary.json")
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
