#
# annotate_sideview_renders.py
#
# For one expname/frame/elev list, take the renders sideview.py already produced (plus the
# PSNR/SSIM warp_to_source.py + score_against_gt.py already scored) and burn the view name
# and its PSNR/SSIM onto the top-left corner of each render, saved as new annotated PNGs.
# Pure numpy/opencv -- no CUDA, no model loading, safe to run on a login node. Assumes
# sideview.py --save_depth --save_meta, warp_to_source.py, and score_against_gt.py have
# already been run for this expname at every requested elev.
#
# Usage:
#   python debugtools/annotate_sideview_renders.py --expname endonerf/cutting-depthreg-aniso1e5-depth002 --frame_idx 40 --elev 10 20 30
#

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))  # project root, for `from utils...`/`from warp_to_source...`
sys.path.insert(0, _here)  # this file's own dir, for the sibling `compare_experiments` import

import argparse

import cv2

from utils.image_utils import locate_sideview_render
from warp_to_source import resolve_iteration
from compare_experiments import load_per_frame

VIEWS = ["central", "left", "right", "up", "down"]


DISPLAY_NAME = {"central": "on traj"}  # display label only -- lookups elsewhere still key on "central";
                                        # avoid a literal hyphen, cv2's Hershey font renders "-" oversized


def annotate(image, view_name, psnr, ssim):
    image = image.copy()
    psnr_str = f"{psnr:.2f}" if psnr is not None else "N/A"
    ssim_str = f"{ssim:.4f}" if ssim is not None else "N/A"
    lines = [f"View: {DISPLAY_NAME.get(view_name, view_name)}", f"PSNR: {psnr_str}", f"SSIM: {ssim_str}"]
    for i, line in enumerate(lines):
        y = 35 + i * 38
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description="Annotate existing sideview renders with view name + PSNR/SSIM")
    parser.add_argument("--expname", required=True, type=str, help="e.g. endonerf/cutting-depthreg-aniso1e5-depth002")
    parser.add_argument("--frame_idx", required=True, type=int)
    parser.add_argument("--elev", required=True, type=float, nargs="+")
    parser.add_argument("--views", nargs="+", default=VIEWS, help=f"default: {VIEWS} (includes central)")
    parser.add_argument("--iteration", type=int, default=None, help="auto-detected if there's only one ours_<iteration> dir")
    parser.add_argument("--out_dir", default=None, help="default: output/<expname>/sideview/ours_<iter>/elev<E>/annotated/")
    args = parser.parse_args()

    sideview_root = os.path.join("output", args.expname, "sideview")
    iteration = resolve_iteration(sideview_root, args.iteration)
    sideview_dir = os.path.join(sideview_root, f"ours_{iteration}")

    # "central" always renders at elev_deg=0 regardless of the requested elev; every other
    # requested view renders at whatever elev is currently being processed below
    saved = 0
    for elev in args.elev:
        metrics = load_per_frame(args.expname, elev, args.views, iteration)

        for view_name in args.views:
            view_elev = 0.0 if view_name == "central" else elev
            render_path, view_elev_dir = locate_sideview_render(sideview_dir, args.frame_idx, view_name, view_elev)
            if render_path is None:
                print(f"skipping {view_name} @ elev={elev:.0f}: no render found for frame {args.frame_idx}")
                continue

            image = cv2.imread(render_path)
            if image is None:
                print(f"skipping {view_name} @ elev={elev:.0f}: failed to read {render_path}")
                continue

            entry = metrics.get((view_name, args.frame_idx))
            psnr = entry.get("psnr") if entry else None
            ssim = entry.get("ssim") if entry else None
            annotated = annotate(image, view_name, psnr, ssim)

            out_dir = args.out_dir or os.path.join(sideview_dir, f"elev{elev:.0f}", "annotated")
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.basename(render_path).replace("render_", "annotated_")
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, annotated)
            print(f"saved {out_path} (psnr={psnr}, ssim={ssim})")
            saved += 1

    if saved == 0:
        raise FileNotFoundError(f"no renders found for frame {args.frame_idx} in {sideview_dir} -- check --expname/--frame_idx/--elev")


if __name__ == "__main__":
    main()
