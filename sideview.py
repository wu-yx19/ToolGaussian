#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Updated by Y. W. 2026
#

import os
import sys
import json
from os import makedirs
import numpy as np
import cv2
from tqdm import tqdm

import torch
from scene.gaussian_renderer import GaussianModel, render
from scene import Scene

from utils.general_utils import format_output, resolve_expname_paths, set_seed
from utils.image_utils import to8b, add_title, concat_with_title
from utils.graphics_utils import process_view, warp_view

from argparse import ArgumentParser
from arguments import ModelHiddenParams, ModelParams, PipelineParams, SideviewParams, get_combined_args, merge_hparams

def get_view_offsets(elev):
    return {
        "central": {"azim": 0, "elev": 0},
        "left": {"azim": 0, "elev": elev},
        "right": {"azim": 180, "elev": elev},
        "up": {"azim": 90, "elev": elev},
        "down": {"azim": 270, "elev": elev},
    }
# render one or more views of a single frame, each relative to that frame's original camera pose
def render_frame_views(
    model_path : str,
    iteration : int,
    view,
    gaussians,
    pipelineParam,
    background, # tensor
    no_fine, # coarse
    frame_idx,
    elev, # elevation magnitude these view_offsets were built from, only used to namespace the concat output
    view_offsets,
    sideviewParam, # extracted SideviewParams group -- concat/save_depth/save_opacity/save_meta/warp_mode
):
    depth = view.original_depth
    depth = depth.cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
    valid_depth = depth[depth > 0]
    if valid_depth.size == 0:
        raise ValueError(f"frame {frame_idx} has no valid (positive) depth values")
    distance = float(np.median(valid_depth))

    out_path = os.path.join(model_path, "sideview", "ours_{}".format(iteration))
    # every view whose own elev_deg matches the requested elev (i.e. everything except
    # "central", which is always elev_deg=0) reuses this same dir -- computed once here
    # instead of re-joining out_path per view
    outer_elev_dir = os.path.join(out_path, f"elev{elev:.0f}")

    stage = "coarse" if no_fine else "fine"

    warp_enabled = sideviewParam.warp_mode != "off"
    if warp_enabled:
        # source color/depth to forward-warp into every offset view below, built once per frame.
        # warp_view needs depth as 2D (H, W); original_depth may carry a trailing channel dim.
        gt_image = np.ascontiguousarray(to8b(view.original_image.cpu()).transpose(1, 2, 0))  # CHW -> HWC
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
        gt_depth = np.squeeze(depth)

        # view.mask: True/1 = valid tissue, False/0 = tool (see scene/datasets.py, "1 - mask/255")
        tool_mask = view.mask
        tool_mask = tool_mask.cpu().numpy() if torch.is_tensor(tool_mask) else np.asarray(tool_mask)
        valid_mask = np.squeeze(tool_mask).astype(bool)

        if sideviewParam.warp_mode in ("render", "gt_fill"):
            # the model's own render at the original (un-offset) pose, used to fill tool-masked
            # gaps in the GT (gt_fill) or to replace GT entirely (render)
            orig_rendering = render(view, gaussians, pipelineParam, background, stage=stage)
            render_image_orig = np.ascontiguousarray(to8b(orig_rendering["render"].cpu()).transpose(1, 2, 0))
            render_image_orig = cv2.cvtColor(render_image_orig, cv2.COLOR_RGB2BGR)
            render_depth_orig = orig_rendering["depth"].cpu().squeeze().numpy().astype(np.float32)

        if sideviewParam.warp_mode == "gt":
            # tool-masked pixels are left invalid rather than trusting whatever raw depth
            # happens to be stored there (GT depth is unreliable/missing under the tool)
            warp_src_image = gt_image
            warp_src_depth = np.where(valid_mask, gt_depth, 0)
        elif sideviewParam.warp_mode == "gt_fill":
            warp_src_image = np.where(valid_mask[..., None], gt_image, render_image_orig)
            warp_src_depth = np.where(valid_mask, gt_depth, render_depth_orig)
        elif sideviewParam.warp_mode == "render":
            warp_src_image = render_image_orig
            warp_src_depth = render_depth_orig
        else:
            raise ValueError(f"unknown warp_mode {sideviewParam.warp_mode!r} (expected off, gt, gt_fill, or render)")

    render_images = []
    depth_images = []
    opacity_images = []
    warped_images = []
    for view_name, offset in view_offsets.items():

        azim_deg = offset["azim"]
        elev_deg = offset["elev"]

        view_elev_dir = outer_elev_dir if elev_deg == elev else os.path.join(out_path, f"elev{elev_deg:.0f}")
        image_path = os.path.join(view_elev_dir, "renders")
        makedirs(image_path, exist_ok=True)
        if sideviewParam.save_depth:
            depth_path = os.path.join(view_elev_dir, "depth")
            makedirs(depth_path, exist_ok=True)
        if sideviewParam.save_opacity:
            opacity_path = os.path.join(view_elev_dir, "opacity")
            makedirs(opacity_path, exist_ok=True)
        if warp_enabled:
            warp_path = os.path.join(view_elev_dir, "warped")
            makedirs(warp_path, exist_ok=True)

        view_new = process_view(view, azim_deg, elev_deg, distance)

        rendering = render(view_new, gaussians, pipelineParam, background, stage=stage)

        # frame_idx is included so per-view files don't collide across frames that
        # happen to round to the same median distance (e.g. when using --frame_stride)
        suffix = f"frame{frame_idx}_{view_name}_azim{azim_deg:.0f}_elev{elev_deg:.0f}_dist{distance:.0f}"

        # individual files are saved untitled -- they're meant for quantitative rendering-quality
        # comparisons, so no burned-in text; the concat figures below get view-name labels instead
        render_image = np.ascontiguousarray(to8b(rendering["render"].cpu()).transpose(1, 2, 0))  # CHW -> HWC
        render_image = cv2.cvtColor(render_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(image_path, f"render_{suffix}.png"), render_image)
        render_images.append(add_title(render_image, view_name))

        if sideviewParam.save_depth:
            raw_depth = rendering["depth"].cpu().squeeze().numpy().astype(np.float32)

            if sideviewParam.save_meta:
                # raw z-buffer depth (float32) + the exact camera params used, so tools like
                # warp_to_source.py can reproject this render without relying on the lossy 8-bit
                # visualization PNG or on re-deriving azim/elev/dist from rounded filenames
                np.save(os.path.join(depth_path, f"depth_{suffix}.npy"), raw_depth)
                metadata = {
                    "frame_idx": frame_idx,
                    "view_name": view_name,
                    "azim": azim_deg,
                    "elev": elev_deg,
                    "dist": distance,
                    "stage": stage,
                    "model_path": model_path,
                    "iteration": iteration,
                    "width": view.image_width,
                    "height": view.image_height,
                    "FoVx": view.FoVx,
                    "FoVy": view.FoVy,
                    "orig_R": np.asarray(view.R).tolist(),
                    "orig_T": np.asarray(view.T).tolist(),
                    "offset_R": np.asarray(view_new.R).tolist(),
                    "offset_T": np.asarray(view_new.T).tolist(),
                }
                with open(os.path.join(depth_path, f"depth_{suffix}.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

            render_depth = np.clip(raw_depth, 0, 255).astype(np.uint8)
            render_depth = cv2.cvtColor(render_depth, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(depth_path, f"depth_{suffix}.png"), render_depth)
            depth_images.append(add_title(render_depth, view_name))

        if sideviewParam.save_opacity:
            # alpha is accumulated per-pixel opacity in [0, 1]; match the viridis/vmin=0/vmax=1
            # visualization used in debugtools/check_accum_alpha.py
            render_opacity = np.clip(rendering["alpha"].cpu().squeeze().numpy() * 255, 0, 255).astype(np.uint8)
            render_opacity = cv2.applyColorMap(render_opacity, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(os.path.join(opacity_path, f"opacity_{suffix}.png"), render_opacity)
            opacity_images.append(add_title(render_opacity, view_name))

        if warp_enabled:
            # forward-warp the frame's source image+depth (per warp_mode) into this offset
            # camera's pose, so it can be compared against the actual render above
            warped, warp_valid = warp_view(
                warp_src_image, warp_src_depth, view.R, view.T, view_new.R, view_new.T,
                view.FoVx, view.FoVy, view.image_width, view.image_height,
            )
            cv2.imwrite(os.path.join(warp_path, f"warped_{suffix}.png"), warped)
            cv2.imwrite(os.path.join(warp_path, f"warped_{suffix}_mask.png"), (warp_valid * 255).astype(np.uint8))
            warped_images.append(add_title(warped, view_name))

    if sideviewParam.concat:
        model_name = os.sep.join(os.path.normpath(model_path).split(os.sep)[-2:])
        azims = ",".join(f"{offset['azim']:.0f}" for offset in view_offsets.values())
        concat_title = f"{model_name} frame={frame_idx} elev={elev:.0f} azim={{{azims}}}"

        # placed under outer_elev_dir (not out_path root) even though "central" panels are
        # sourced from elev0/ -- outer_elev_dir is what this whole render pass corresponds to
        makedirs(outer_elev_dir, exist_ok=True)

        concat_path = os.path.join(outer_elev_dir, f"frame{frame_idx}_elev{elev:.0f}_concat.png")
        concat_with_title(render_images, concat_title, concat_path)
        if sideviewParam.save_depth:
            depth_concat_path = os.path.join(outer_elev_dir, f"frame{frame_idx}_elev{elev:.0f}_concat_depth.png")
            concat_with_title(depth_images, concat_title, depth_concat_path)
        if sideviewParam.save_opacity:
            opacity_concat_path = os.path.join(outer_elev_dir, f"frame{frame_idx}_elev{elev:.0f}_concat_opacity.png")
            concat_with_title(opacity_images, concat_title, opacity_concat_path)
        if warp_enabled:
            warped_concat_path = os.path.join(outer_elev_dir, f"frame{frame_idx}_elev{elev:.0f}_concat_warped.png")
            concat_with_title(warped_images, concat_title, warped_concat_path)


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters (single frame, one or more views)")
    modelParam = ModelParams()
    pipelineParam = PipelineParams()
    modelHiddenParam = ModelHiddenParams()
    sideviewParam = SideviewParams()
    sideviewParam.frame_idxs = [0] # sideview.py renders frame 0 by default, unlike render.py

    modelParam.register(parser, set_default_none=True)
    # set_default_none=True here too: without it, get_combined_args only lets a saved cfg_args
    # value through when the cmdline value is None -- since these fields always get concrete
    # argparse defaults otherwise, the actual trained architecture/pipeline settings get silently
    # replaced by defaults for any checkpoint lacking a matching arguments/<expname>.py override
    pipelineParam.register(parser, set_default_none=True)
    modelHiddenParam.register(parser, set_default_none=True)
    sideviewParam.register(parser)

    parser.add_argument("--configs", type=str)
    parser.add_argument("--iteration", default=-1, type=int) # load iteraton, default -1 -> maximum iteration
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--expname", type=str, default="")

    # configs > cmdline > model cfg_args > default
    args = parser.parse_args(sys.argv[1:])
    configs_auto_derived = bool(args.expname and not args.configs)
    args = resolve_expname_paths(args)
    args = get_combined_args(args) # read modelpath args, overwrite with cmdline
    if args.configs and (not configs_auto_derived or os.path.isfile(args.configs)):
        # overwrite with configs -- but an auto-derived --configs (from --expname) is a guess,
        # not a user request, so a missing file there just means "no override, use cfg_args
        # alone" instead of a crash; an explicitly-passed --configs that's missing still errors
        import mmcv
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    elif configs_auto_derived:
        print(f"No config file at {args.configs} for --expname {args.expname}; using only the saved cfg_args (no --configs overrides applied)")

    format_output(args.quiet)
    set_seed(0) # Initialize random seed
    torch.cuda.set_device(torch.device("cuda:0"))

    modelParam = modelParam.extract(args)
    modelHiddenParam = modelHiddenParam.extract(args)
    pipelineParam = pipelineParam.extract(args)
    sideviewParam = sideviewParam.extract(args)

    if sideviewParam.warp_mode not in ("off", "gt", "gt_fill", "render"):
        raise ValueError(f"--warp_mode must be one of off, gt, gt_fill, render (got {sideviewParam.warp_mode!r})")

    with torch.no_grad():
        gaussians = GaussianModel(modelParam.sh_degree, modelHiddenParam)
        scene = Scene(
            modelParam,
            gaussians,
            load_iteration=args.iteration,
            load_coarse=modelParam.no_fine,
        )

        bg_color = [1, 1, 1] if modelParam.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if sideviewParam.frame_stride and "--frame_idxs" in sys.argv:
            print("Warning: --frame_stride and --frame_idxs both given; --frame_stride takes precedence, --frame_idxs is ignored")

        if sideviewParam.sideview_on_test:
            # view.uid is the frame's true index in the full sequence (set at dataset-load
            # time), which still matches the video set's own gt/masks output naming (the
            # video split covers every frame in order), so score_against_gt.py's GT
            # lookup by frame_idx keeps working even though frames now come from the test split
            frame_views = [(view.uid, view) for view in scene.getTestViews()]
        else:
            video_views = scene.getVideoViews()
            frame_idxs = (
                list(range(0, len(video_views), int(sideviewParam.frame_stride)))
                if sideviewParam.frame_stride else sideviewParam.frame_idxs
            )
            frame_views = [(frame_idx, video_views[frame_idx]) for frame_idx in frame_idxs]

        print("Rendering ", args.model_path, f"(frames {[f for f, _ in frame_views]}, views: {sideviewParam.views}, elevs: {sideviewParam.elev})")

        # loop over elevs on the already-loaded scene, instead of reloading the model once per elev
        for elev in sideviewParam.elev:
            all_view_offsets = get_view_offsets(elev)
            view_offsets = {viewname: all_view_offsets[viewname] for viewname in sideviewParam.views}
            for frame_idx, view in tqdm(frame_views, desc=f"Rendering sideviews (elev={elev})"):
                render_frame_views(
                    modelParam.model_path,
                    scene.loaded_iter,
                    view,
                    scene.gaussians,
                    pipelineParam,
                    background,
                    modelParam.no_fine,
                    frame_idx,
                    elev,
                    view_offsets,
                    sideviewParam,
                )

    print("\nRendering complete")
