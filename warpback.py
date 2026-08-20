#
# warpback.py
#
# Batch tool: given an expname and one or more elevs, warp every view's render
# (at every frame found) back into the frame's original camera via
# utils.graphics_utils.warp_view (a forward, z-buffered point splat). Mirrors
# sideview.py's own output layout: "central" always comes from elev0/ regardless
# of the requested elev, and each view's warped output lands in its own
# elev<E>/warpback/ dir (sibling to renders/, depth/, etc.).
#
# Requires the renders to have been produced by `sideview.py --save_depth
# --save_meta`, which writes the raw float32 depth (.npy) and the exact camera
# parameters (.json) this tool needs alongside the usual 8-bit visualization
# PNGs. Pure numpy/opencv -- no CUDA, no model loading, safe to run on a login
# node.
#
# Usage:
#   python warpback.py --expname endonerf/pulling --elev 10 20 30
#   python warpback.py --expname endonerf/pulling --iteration 6000 --elev 20
#

import os
import re
import glob
import json
import argparse

import numpy as np
import cv2

from utils.graphics_utils import warp_view
from utils.image_utils import sideview_view_elevs, locate_sideview_render


def load_sidecar(image_path):
    # renders/render_<suffix>.png -> depth/depth_<suffix>.{npy,json}
    renders_dir, image_name = os.path.split(image_path)
    depth_dir = os.path.join(os.path.dirname(renders_dir), "depth")
    suffix = image_name[len("render_"):-len(".png")]
    depth_path = os.path.join(depth_dir, f"depth_{suffix}.npy")
    meta_path = os.path.join(depth_dir, f"depth_{suffix}.json")
    if not os.path.isfile(depth_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"missing sidecar for {image_path}: expected {depth_path} and {meta_path} "
            "(re-render with `sideview.py --save_depth --save_meta`)"
        )
    depth = np.load(depth_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return depth, meta


def discover_frame_indices(renders_dir):
    frame_idxs = set()
    for path in glob.glob(os.path.join(renders_dir, "render_frame*_*.png")):
        m = re.match(r"render_frame(\d+)_", os.path.basename(path))
        if m:
            frame_idxs.add(int(m.group(1)))
    return sorted(frame_idxs)


def resolve_iteration(sideview_root, iteration):
    if iteration is not None:
        return iteration
    candidates = sorted(int(m.group(1)) for d in glob.glob(os.path.join(sideview_root, "ours_*"))
                         if (m := re.fullmatch(r"ours_(\d+)", os.path.basename(d))))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no ours_<iteration> dirs found under {sideview_root}")
    raise ValueError(f"multiple ours_<iteration> dirs found under {sideview_root} ({candidates}); pass --iteration to pick one")


def warp_one(render_path, out_dir):
    depth, meta = load_sidecar(render_path)
    image = cv2.imread(render_path)
    if image is None:
        raise FileNotFoundError(render_path)
    if image.shape[:2] != depth.shape:
        raise ValueError(f"image shape {image.shape[:2]} != depth shape {depth.shape}")

    warped, mask = warp_view(
        image, depth,
        meta["offset_R"], meta["offset_T"],  # src: the camera that actually captured this render
        meta["orig_R"], meta["orig_T"],      # tgt: warp back to the frame's original camera
        meta["FoVx"], meta["FoVy"], meta["width"], meta["height"],
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(render_path).replace("render_", "warpback_"))
    cv2.imwrite(out_path, warped)
    mask_path = os.path.splitext(out_path)[0] + "_mask.png"
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    return out_path, float(mask.mean())


def main():
    parser = argparse.ArgumentParser(description="Warp sideview.py renders back to their original camera, for every view/frame at the given elevs")
    parser.add_argument("--expname", required=True, type=str, help="e.g. endonerf/pulling -> output/endonerf/pulling")
    parser.add_argument("--iteration", type=int, default=None, help="the ours_<iteration> checkpoint sideview.py rendered from (auto-detected if there's only one)")
    parser.add_argument("--elev", required=True, type=float, nargs="+", help="one or more elevations to process")
    args = parser.parse_args()

    sideview_root = os.path.join("output", args.expname, "sideview")
    iteration = resolve_iteration(sideview_root, args.iteration)
    sideview_dir = os.path.join(sideview_root, f"ours_{iteration}")

    total, skipped = 0, 0
    for elev in args.elev:
        frame_idxs_by_dir = {}
        for view_name, view_elev in sideview_view_elevs(elev):
            view_elev_dir = os.path.join(sideview_dir, f"elev{view_elev:.0f}")
            renders_dir = os.path.join(view_elev_dir, "renders")
            if view_elev_dir not in frame_idxs_by_dir:
                frame_idxs_by_dir[view_elev_dir] = discover_frame_indices(renders_dir) if os.path.isdir(renders_dir) else []

            for frame_idx in frame_idxs_by_dir[view_elev_dir]:
                render_path, _ = locate_sideview_render(sideview_dir, frame_idx, view_name, view_elev)
                if render_path is None:
                    continue
                out_dir = os.path.join(view_elev_dir, "warpback")
                try:
                    out_path, coverage = warp_one(render_path, out_dir)
                except FileNotFoundError as e:
                    print(f"skipping {render_path}: {e}")
                    skipped += 1
                    continue
                print(f"wrote {out_path} (coverage {coverage:.1%})")
                total += 1

    print(f"warped {total} views ({skipped} skipped, missing --save_meta sidecar)")


if __name__ == "__main__":
    main()
