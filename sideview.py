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
from os import makedirs
import numpy as np
import cv2
from tqdm import tqdm

import torch
from scene.gaussian_renderer import GaussianModel, render
from scene import Scene

from utils.general_utils import format_output, set_seed
from utils.image_utils import to8b, save_with_title, concat_with_title
from utils.graphics_utils import process_view

import argparse
from argparse import ArgumentParser
from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args, merge_hparams

ELEV = 15

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
    view_offsets,
    concat,
):
    depth = view.original_depth
    depth = depth.cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
    valid_depth = depth[depth > 0]
    if valid_depth.size == 0:
        raise ValueError(f"frame {frame_idx} has no valid (positive) depth values")
    distance = float(np.median(valid_depth))

    out_path = os.path.join(model_path, "sideview", "ours_{}".format(iteration))

    stage = "coarse" if no_fine else "fine"

    render_images = []
    for view_name, offset in view_offsets.items():

        azim_deg = offset["azim"]
        elev_deg = offset["elev"]

        elev_dir = f"elev{elev_deg:.0f}"
        image_path = os.path.join(out_path, elev_dir, "renders")
        depth_path = os.path.join(out_path, elev_dir, "depth")
        makedirs(image_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

        view_new = process_view(view, azim_deg, elev_deg, distance)

        rendering = render(view_new, gaussians, pipelineParam, background, stage=stage)

        title = f"view={view_name} azim={azim_deg:.1f} elev={elev_deg:.1f} dist={distance:.1f}"
        suffix = f"{view_name}_azim{azim_deg:.0f}_elev{elev_deg:.0f}_dist{distance:.0f}"

        render_image = np.ascontiguousarray(to8b(rendering["render"].cpu()).transpose(1, 2, 0))  # CHW -> HWC
        render_image = cv2.cvtColor(render_image, cv2.COLOR_RGB2BGR)
        save_with_title(render_image, title, os.path.join(image_path, f"render_{suffix}.png"))
        render_images.append(render_image)

        render_depth = np.clip(rendering["depth"].cpu().squeeze().numpy(), 0, 255).astype(np.uint8)
        save_with_title(render_depth, title, os.path.join(depth_path, f"depth_{suffix}.png"))

    if concat:
        model_name = os.sep.join(os.path.normpath(model_path).split(os.sep)[-2:])
        concat_path = os.path.join(out_path, f"frame{frame_idx}_concat.png")
        concat_with_title(render_images, model_name, concat_path)


if __name__ == "__main__":

    # azim/elev offsets applied on top of the frame's original camera pose
    VIEW_OFFSETS = get_view_offsets(ELEV)

    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters (single frame, one or more views)")
    modelParam = ModelParams()
    pipelineParam = PipelineParams()
    modelHiddenParam = ModelHiddenParams()

    modelParam.register(parser, set_default_none=True)
    pipelineParam.register(parser)
    modelHiddenParam.register(parser)

    parser.add_argument("--configs", type=str)
    parser.add_argument("--iteration", default=-1, type=int) # load iteraton, default -1 -> maximum iteration
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--expname", type=str, default="")

    parser.add_argument("--frame_idxs", nargs="+", type=int, default=[0]) # indices of the frames to render
    parser.add_argument("--frame_stride", type=int, default=None) # if set, render every Nth frame of the video set instead of --frame_idxs
    parser.add_argument("--views", nargs="+", default=["central", "left", "right", "up", "down"], choices=list(VIEW_OFFSETS.keys())) # one or more views, relative to the frame's original pose
    parser.add_argument("--concat", action=argparse.BooleanOptionalAction, default=True) # concat rendered views into one titled figure (--no-concat to disable)

    # configs > cmdline > model cfg_args > default
    args = parser.parse_args(sys.argv[1:])
    if args.expname:
        if not args.model_path:
            args.model_path = os.path.join("./output/", args.expname)
        if not args.configs:
            args.configs = os.path.join("./arguments/", args.expname + ".py")
    args = get_combined_args(args) # read modelpath args, overwrite with cmdline
    if args.configs: # overwrite with configs
        import mmcv
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    format_output(args.quiet)
    set_seed(0) # Initialize random seed
    torch.cuda.set_device(torch.device("cuda:0"))

    modelParam = modelParam.extract(args)
    modelHiddenParam = modelHiddenParam.extract(args)
    pipelineParam = pipelineParam.extract(args)

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

        video_views = scene.getVideoViews()
        view_offsets = {viewname: VIEW_OFFSETS[viewname] for viewname in args.views}
        frame_idxs = list(range(0, len(video_views), args.frame_stride)) if args.frame_stride else args.frame_idxs

        print("Rendering ", args.model_path, f"(frames {frame_idxs}, views: {args.views})")

        for frame_idx in tqdm(frame_idxs, desc="Rendering sideviews"):
            render_frame_views(
                modelParam.model_path,
                scene.loaded_iter,
                video_views[frame_idx],
                scene.gaussians,
                pipelineParam,
                background,
                modelParam.no_fine,
                frame_idx,
                view_offsets,
                args.concat,
            )

    print("\nRendering complete")
