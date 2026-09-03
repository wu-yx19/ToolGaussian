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
from time import time
from tqdm import tqdm

import torch
from scene.gaussian_renderer import GaussianModel, render
from scene import Scene

from utils.general_utils import format_output, set_seed
from utils.graphics_utils import fov2focal, reconstruct_point_cloud
from utils.image_utils import write_images, write_video

from argparse import ArgumentParser, BooleanOptionalAction
from arguments import GroupParams, ModelHiddenParams, ModelParams, PipelineParams, get_combined_args, merge_hparams

# render a set of views
def render_set(
    model_path : str,
    name : str,
    iteration : int,
    views, # list of views
    gaussians,
    pipelineParam,
    background, # tensor
    no_fine, # coarse
    render_test=False,
    reconstruct=False,
):
    image_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gtimage_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gtdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")

    makedirs(image_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gtimage_path, exist_ok=True)
    makedirs(gtdepth_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    render_images = []
    render_depths = []
    gt_images = []
    gt_depths = []
    masks = []

    for idx, view in enumerate(tqdm(views, desc="Rendering Progress ({})".format(name))): # view: Views

        stage = "coarse" if no_fine else "fine"
        rendering = render(view, gaussians, pipelineParam, background, stage=stage)
        render_depths.append(rendering["depth"].cpu())
        render_images.append(rendering["render"].cpu())
        if name in ["train", "test", "video"]:
            gt = view.original_image[0:3, :, :]
            gt_images.append(gt)
            mask = view.mask
            masks.append(mask)
            gt_depth = view.original_depth
            gt_depths.append(gt_depth) # append ground truth

    if render_test: # do a test on the rendering process, not to be confused with rendering the test set
        test_times = 50
        for i in range(test_times):
            for idx, view in enumerate(tqdm(views, desc="Rendering Test Progress ({})".format(name))):
                if idx == 0 and i == 0:
                    time1 = time()
                stage = "coarse" if no_fine else "fine"
                rendering = render(view, gaussians, pipelineParam, background, stage=stage)
        time2 = time()
        print("FPS:", (len(views) - 1) * test_times / (time2 - time1))

    frame_idxs = [view.uid for view in views]

    write_images("rendered", render_images, image_path, frame_idxs)
    write_images("rendered depth", render_depths, depth_path, frame_idxs)
    write_images("ground truth", gt_images, gtimage_path, frame_idxs)
    write_images("ground truth depth", gt_depths, gtdepth_path, frame_idxs)
    write_images("mask", masks, mask_path, frame_idxs)

    write_video(render_images, os.path.join(model_path, name, "ours_{}".format(iteration), "ours_video.mp4"))
    write_video(gt_images, os.path.join(model_path, name, "ours_{}".format(iteration), "gt_video.mp4"))

    if reconstruct: # RDB+D -> point cloud
        focal_x, focal_y = fov2focal(view.FoVx, view.image_width), fov2focal(view.FoVy, view.image_height)
        camera_parameters = (focal_x, focal_y, view.image_width, view.image_height)
        output_frame_folder = os.path.join("reconstruct", name)
        reconstruct_point_cloud(render_images, masks, render_depths, camera_parameters, output_frame_folder)


def render_sets(
    modelParam: GroupParams,
    modelHiddenParam: GroupParams,
    pipelineParam: GroupParams,
    iteration: int,
    skip_train: bool,
    skip_test: bool,
    skip_video: bool,
    reconstruct: bool,
    render_test: bool,
):

    with torch.no_grad():
        gaussians = GaussianModel(modelParam.sh_degree, modelHiddenParam)
        scene = Scene(
            modelParam,
            gaussians,
            load_iteration=iteration, # load
            load_coarse=modelParam.no_fine,
        )

        bg_color = [1, 1, 1] if modelParam.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                modelParam.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainViews(),
                scene.gaussians,
                pipelineParam,
                background,
                modelParam.no_fine,
                reconstruct=False,
            )
        if not skip_test:
            render_set(
                modelParam.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestViews(),
                scene.gaussians,
                pipelineParam,
                background,
                modelParam.no_fine,
                reconstruct=reconstruct,
            )
        if not skip_video:
            render_set(
                modelParam.model_path,
                "video",
                scene.loaded_iter,
                scene.getVideoViews(),
                scene.gaussians,
                pipelineParam,
                background,
                modelParam.no_fine,
                render_test=render_test,
                reconstruct=reconstruct,
            )


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    modelParam = ModelParams()
    pipelineParam = PipelineParams()
    modelHiddenParam = ModelHiddenParams()

    modelParam.register(parser, set_default_none=True)
    pipelineParam.register(parser, set_default_none=True)
    modelHiddenParam.register(parser, set_default_none=True)

    parser.add_argument("--configs", type=str)
    parser.add_argument("--iteration", default=-1, type=int) # load iteraton, default -1 -> maximum iteration
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--expname", type=str, default="")

    parser.add_argument("--skip_train", action=BooleanOptionalAction, default=True)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true") # all Views
    parser.add_argument("--reconstruct", action="store_true") # reconstruct point cloud from RGB-D
    parser.add_argument("--render_test", action="store_true") # test rendering speed on the video set


    # configs > cmdline > model cfg_args > default
    args = parser.parse_args(sys.argv[1:])
    configs_auto_derived = False
    if args.expname:
        if not args.model_path:
            args.model_path = os.path.join("./output/", args.expname)
        if not args.configs:
            args.configs = os.path.join("./arguments/", args.expname + ".py")
            configs_auto_derived = True
    args = get_combined_args(args) # read modelpath args, overwrite with cmdline
    if args.configs and (not configs_auto_derived or os.path.isfile(args.configs)):
        import mmcv
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    elif configs_auto_derived:
        print(f"No config file at {args.configs} for --expname {args.expname}; using only the saved cfg_args (no --configs overrides applied)")

    format_output(args.quiet)
    set_seed(0) # Initialize random seed
    torch.cuda.set_device(torch.device("cuda:0"))

    print("Rendering ", args.model_path)

    render_sets(
        modelParam.extract(args),
        modelHiddenParam.extract(args),
        pipelineParam.extract(args),
        args.iteration,
        args.skip_train,
        args.skip_test,
        args.skip_video,
        args.reconstruct,
        args.render_test,
    )

    print("\nRendering complete")
