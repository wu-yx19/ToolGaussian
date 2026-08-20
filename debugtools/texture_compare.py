#
# texture_compare.py
#
# For every view rendered at a given elev, compare texture richness between the
# render and its corresponding warped output (sideview.py --warp_mode), restricted
# to the warped image's valid (covered) pixels -- so the comparison only reflects
# regions the warp actually produced data for. Mirrors sideview.py's own output
# layout: "central" always comes from elev0/ regardless of the requested elev, and
# both individual heatmaps and the concat montage land under elev<E>/ -- exactly
# where sideview.py itself writes frame<F>_elev<E>_concat*.png. The one exception
# is the stats json, which spans two elev dirs (central's elev0 + this elev<E>) and
# so is written under sideview_dir/texture_compare/ instead of either one alone.
#
# Metrics (utils/image_utils.masked_local_variance / masked_gradient_energy):
#   - local variance: intensity-spread richness in a window (normalized convolution,
#     unbiased even with holes in the mask)
#   - gradient energy: Sobel edge/detail richness (only evaluated where the full
#     3x3 neighborhood is valid)
#
# Usage:
#   python texture_compare.py --sideview_dir output/endonerf/pulling/sideview/ours_6000 \
#                              --elev 20 --frame_idx 0
#

import os
import json
import argparse

import numpy as np
import cv2

from utils.image_utils import compute_texture_density, load_gray, texture_heatmap, add_title, concat_with_title, sideview_view_elevs, locate_sideview_render


def default_mask_path(warped_path):
    root, ext = os.path.splitext(warped_path)
    return f"{root}_mask{ext}"


def summarize(values, valid):
    v = values[valid]
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(v.size)}


def main():
    parser = argparse.ArgumentParser(description="Compare texture richness between renders and their warped counterparts, across all 5 sideview.py views at one elev")
    parser.add_argument("--sideview_dir", required=True, type=str, help="the 'ours_<iteration>' dir sideview.py wrote (parent of elev<E> folders)")
    parser.add_argument("--elev", required=True, type=float, help="elevation to inspect (central still comes from elev0/, per sideview.py)")
    parser.add_argument("--frame_idx", required=True, type=int, help="which frame's set of views to process")
    parser.add_argument("--window", type=int, default=15, help="local-variance window size in pixels")
    parser.add_argument("--coverage_thresh", type=float, default=0.5, help="min fraction of a variance window that must be valid to trust it")
    parser.add_argument("--grad_ksize", type=int, default=3, help="Sobel kernel size for the gradient metric (1, 3, 5, or 7); larger = coarser-scale edges, less noise-sensitive")
    args = parser.parse_args()

    views = []
    for view_name, view_elev in sideview_view_elevs(args.elev):
        render_path, view_elev_dir = locate_sideview_render(args.sideview_dir, args.frame_idx, view_name, view_elev)
        if render_path is None:
            print(f"skipping {view_name}: no render found (expected in elev{view_elev:.0f}/renders/)")
            continue
        warped_path = os.path.join(view_elev_dir, "warped", f"warped_{os.path.basename(render_path)[len('render_'):-len('.png')]}.png")
        mask_path = default_mask_path(warped_path)
        if not (os.path.isfile(warped_path) and os.path.isfile(mask_path)):
            print(f"skipping {view_name}: no warped output found ({warped_path})")
            continue
        views.append((view_name, render_path, warped_path, mask_path, view_elev_dir))
    if not views:
        raise FileNotFoundError(f"no render+warped pairs found for frame {args.frame_idx} under {args.sideview_dir}")

    # pass 1: compute both images' texture maps for every view
    per_view = {}
    for view_name, render_path, warped_path, mask_path, _ in views:
        render_gray = load_gray(render_path)
        warped_gray = load_gray(warped_path)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(mask_path)
        mask = mask_img > 127

        if render_gray.shape != warped_gray.shape or render_gray.shape != mask.shape:
            raise ValueError(f"{view_name}: shape mismatch render {render_gray.shape}, warped {warped_gray.shape}, mask {mask.shape}")

        # the render has no holes of its own; gating both sides by the warped image's mask
        # keeps the comparison apples-to-apples (same pixels scored for both images)
        full_mask = np.ones_like(mask, dtype=bool)
        render_var, _, render_grad, _ = compute_texture_density(render_gray, full_mask, args.window, args.coverage_thresh, args.grad_ksize)
        warped_var, valid_var, warped_grad, valid_grad = compute_texture_density(warped_gray, mask, args.window, args.coverage_thresh, args.grad_ksize)

        per_view[view_name] = {
            "render_var": render_var, "warped_var": warped_var, "valid_var": valid_var,
            "render_grad": render_grad, "warped_grad": warped_grad, "valid_grad": valid_grad,
            "mask_coverage": float(mask.mean()),
        }

    # shared color scale across every view + both images, so heatmaps are comparable
    # in the concat montage instead of each view independently rescaling itself
    all_var = np.concatenate([np.concatenate([v["render_var"][v["valid_var"]], v["warped_var"][v["valid_var"]]]) for v in per_view.values() if v["valid_var"].any()])
    all_grad = np.concatenate([np.concatenate([v["render_grad"][v["valid_grad"]], v["warped_grad"][v["valid_grad"]]]) for v in per_view.values() if v["valid_grad"].any()])
    var_vmax = np.percentile(all_var, 99) if all_var.size else 1.0
    grad_vmax = np.percentile(all_grad, 99) if all_grad.size else 1.0

    # pass 2: per-view stats + individual heatmaps + panels for the concat montages
    stats = {"elev": args.elev, "frame_idx": args.frame_idx, "window": args.window, "grad_ksize": args.grad_ksize, "views": {}}
    var_panels, grad_panels = [], []
    for view_name, render_path, warped_path, _, view_elev_dir in views:
        v = per_view[view_name]

        stats["views"][view_name] = {
            "mask_coverage": v["mask_coverage"],
            "variance": {"render": summarize(v["render_var"], v["valid_var"]), "warped": summarize(v["warped_var"], v["valid_var"])},
            "gradient_energy": {"render": summarize(v["render_grad"], v["valid_grad"]), "warped": summarize(v["warped_grad"], v["valid_grad"])},
        }
        for key in ("variance", "gradient_energy"):
            r, w = stats["views"][view_name][key]["render"], stats["views"][view_name][key]["warped"]
            if r and w:
                stats["views"][view_name][key]["ratio_warped_over_render"] = w["mean"] / max(r["mean"], 1e-6)

        render_base = os.path.splitext(os.path.basename(render_path))[0]
        warped_base = os.path.splitext(os.path.basename(warped_path))[0]

        render_var_map = texture_heatmap(v["render_var"], v["valid_var"], var_vmax)
        warped_var_map = texture_heatmap(v["warped_var"], v["valid_var"], var_vmax)
        render_grad_map = texture_heatmap(v["render_grad"], v["valid_grad"], grad_vmax)
        warped_grad_map = texture_heatmap(v["warped_grad"], v["valid_grad"], grad_vmax)

        # individual heatmaps sit next to their source render, same as sideview.py's own
        # per-view artifacts (renders/, depth/, opacity/, warped/ all live under elev<E>/)
        density_dir = os.path.join(view_elev_dir, "texture_density")
        os.makedirs(density_dir, exist_ok=True)
        cv2.imwrite(os.path.join(density_dir, f"texture_density_variance_{render_base}.png"), render_var_map)
        cv2.imwrite(os.path.join(density_dir, f"texture_density_variance_{warped_base}.png"), warped_var_map)
        cv2.imwrite(os.path.join(density_dir, f"texture_density_gradient_k{args.grad_ksize}_{render_base}.png"), render_grad_map)
        cv2.imwrite(os.path.join(density_dir, f"texture_density_gradient_k{args.grad_ksize}_{warped_base}.png"), warped_grad_map)

        var_panels += [add_title(render_var_map, f"{view_name} render"), add_title(warped_var_map, f"{view_name} warped")]
        grad_panels += [add_title(render_grad_map, f"{view_name} render"), add_title(warped_grad_map, f"{view_name} warped")]

    # concat montages + stats: sideview_dir root, matching where sideview.py writes its own
    # frame<F>_elev<E>_concat*.png (under this call's own elev<E> dir, not the sideview_dir root)
    outer_elev_dir = os.path.join(args.sideview_dir, f"elev{args.elev:.0f}")
    os.makedirs(outer_elev_dir, exist_ok=True)
    concat_title = f"frame={args.frame_idx} elev={args.elev:.0f} grad_ksize={args.grad_ksize}"
    var_concat_path = os.path.join(outer_elev_dir, f"frame{args.frame_idx}_elev{args.elev:.0f}_concat_texture_compare_variance.png")
    grad_concat_path = os.path.join(outer_elev_dir, f"frame{args.frame_idx}_elev{args.elev:.0f}_concat_texture_compare_gradient_k{args.grad_ksize}.png")
    concat_with_title(var_panels, concat_title, var_concat_path)
    concat_with_title(grad_panels, concat_title, grad_concat_path)

    # the stats json spans two elev dirs (central's elev0 + this elev<E>), so it goes in its
    # own top-level folder rather than either elev dir specifically
    stats_dir = os.path.join(args.sideview_dir, "texture_compare")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"texture_compare_frame{args.frame_idx}_elev{args.elev:.0f}_gradk{args.grad_ksize}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")
    print(f"wrote {var_concat_path}")
    print(f"wrote {grad_concat_path}")


if __name__ == "__main__":
    main()
