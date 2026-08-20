#
# texture_density.py
#
# Visualize per-pixel texture density of the renders (no mask -- the full render,
# not gated by any warped/tool mask) for every view at a given elev, mirroring
# sideview.py's own output layout: "central" always comes from elev0/ regardless
# of the requested elev, and both individual heatmaps and the concat montage land
# under elev<E>/ -- exactly where sideview.py itself writes frame<F>_elev<E>_concat*.png.
#
# Two metrics (utils/image_utils.masked_local_variance / masked_gradient_energy):
#   - variance: windowed intensity-spread richness
#   - gradient: Sobel edge/detail richness
#
# Usage:
#   python texture_density.py --sideview_dir output/endonerf/pulling/sideview/ours_6000 \
#                              --elev 20 --frame_idx 0
#

import os
import argparse

import numpy as np
import cv2

from utils.image_utils import compute_texture_density, load_gray, texture_heatmap, add_title, concat_with_title, sideview_view_elevs, locate_sideview_render


def main():
    parser = argparse.ArgumentParser(description="Visualize render texture density (variance + gradient energy), unmasked, across all 5 sideview.py views at one elev")
    parser.add_argument("--sideview_dir", required=True, type=str, help="the 'ours_<iteration>' dir sideview.py wrote (parent of elev<E> folders)")
    parser.add_argument("--elev", required=True, type=float, help="elevation to inspect (central still comes from elev0/, per sideview.py)")
    parser.add_argument("--frame_idx", required=True, type=int, help="which frame's set of views to process")
    parser.add_argument("--window", type=int, default=15, help="local-variance window size in pixels")
    parser.add_argument("--grad_ksize", type=int, default=3, help="Sobel kernel size for the gradient metric (1, 3, 5, or 7); larger = coarser-scale edges, less noise-sensitive")
    args = parser.parse_args()

    views = []
    for view_name, view_elev in sideview_view_elevs(args.elev):
        render_path, view_elev_dir = locate_sideview_render(args.sideview_dir, args.frame_idx, view_name, view_elev)
        if render_path is None:
            print(f"skipping {view_name}: no render found (expected in elev{view_elev:.0f}/renders/)")
            continue
        views.append((view_name, render_path, view_elev_dir))
    if not views:
        raise FileNotFoundError(f"no renders found for frame {args.frame_idx} under {args.sideview_dir}")

    # pass 1: texture maps for every view, no mask (the whole image is valid)
    per_view = {}
    for view_name, render_path, _ in views:
        gray = load_gray(render_path)
        full_mask = np.ones(gray.shape, dtype=bool)
        variance, valid_var, gradient, valid_grad = compute_texture_density(gray, full_mask, args.window, grad_ksize=args.grad_ksize)
        per_view[view_name] = {"variance": variance, "valid_var": valid_var, "gradient": gradient, "valid_grad": valid_grad}

    # shared color scale across all views, so the concat montage is directly comparable
    var_vmax = np.percentile(np.concatenate([v["variance"][v["valid_var"]] for v in per_view.values()]), 99)
    grad_vmax = np.percentile(np.concatenate([v["gradient"][v["valid_grad"]] for v in per_view.values()]), 99)

    var_panels, grad_panels = [], []
    for view_name, render_path, view_elev_dir in views:
        v = per_view[view_name]
        base = os.path.splitext(os.path.basename(render_path))[0]

        var_map = texture_heatmap(v["variance"], v["valid_var"], var_vmax)
        grad_map = texture_heatmap(v["gradient"], v["valid_grad"], grad_vmax)

        # individual heatmaps sit next to their source render, same as sideview.py's own
        # per-view artifacts (renders/, depth/, opacity/ all live under elev<E>/)
        density_dir = os.path.join(view_elev_dir, "texture_density")
        os.makedirs(density_dir, exist_ok=True)
        cv2.imwrite(os.path.join(density_dir, f"texture_density_variance_{base}.png"), var_map)
        cv2.imwrite(os.path.join(density_dir, f"texture_density_gradient_k{args.grad_ksize}_{base}.png"), grad_map)

        var_panels.append(add_title(var_map, view_name))
        grad_panels.append(add_title(grad_map, view_name))

    # concat montage: under this call's own elev<E> dir (not the sideview_dir root), matching
    # sideview.py's own frame<F>_elev<E>_concat*.png placement
    outer_elev_dir = os.path.join(args.sideview_dir, f"elev{args.elev:.0f}")
    os.makedirs(outer_elev_dir, exist_ok=True)
    concat_title = f"render texture density (unmasked) frame={args.frame_idx} elev={args.elev:.0f} grad_ksize={args.grad_ksize}"
    var_concat_path = os.path.join(outer_elev_dir, f"frame{args.frame_idx}_elev{args.elev:.0f}_concat_texture_density_variance.png")
    grad_concat_path = os.path.join(outer_elev_dir, f"frame{args.frame_idx}_elev{args.elev:.0f}_concat_texture_density_gradient_k{args.grad_ksize}.png")
    concat_with_title(var_panels, concat_title, var_concat_path)
    concat_with_title(grad_panels, concat_title, grad_concat_path)

    print(f"wrote {var_concat_path}")
    print(f"wrote {grad_concat_path}")


if __name__ == "__main__":
    main()
