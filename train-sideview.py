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
# Same training loop as train.py, plus a side-view smoothness regularizer:
# each iteration (every sideview_reg_interval iters) we synthesize a side
# view of the current training camera (reusing render-sideview.py's pose
# synthesis) and penalize its TV-loss, to discourage the streaky/needle
# Gaussians that only show up when viewed off-axis from training cameras.
#

import sys
import importlib.util
from pathlib import Path
from argparse import ArgumentParser, Namespace
from random import randint, choice
import os
import lpips
import numpy as np
import torch
from tqdm import tqdm

from utils.general_utils import set_seed, set_seed_train, format_output, training_report
from utils.graphics_utils import render_training_image, process_view
from utils.eval_utils import psnr
from utils.loss_utils import TV_loss, anisotropy_loss, l1_loss, lpips_loss, ssim
from utils.time_utils import Timer

from scene import GaussianModel, Scene
from scene.gaussian_renderer import network_gui, render
from arguments import ModelHiddenParams, ModelParams, PipelineParams, OptimizationParams, merge_hparams
from torchmetrics.functional.regression import pearson_corrcoef

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

elev = 15
VIEW_OFFSETS = {
    "left": {"azim": 0, "elev": elev},
    "right": {"azim": 180, "elev": elev},
    "up": {"azim": 90, "elev": elev},
    "down": {"azim": 270, "elev": elev},
}

def render_side_view(viewpoint_cam, gaussians, pipelineParam, background, stage):
    # pick one of the non-central VIEW_OFFSETS (left/right/up/down) and render it
    depth = viewpoint_cam.original_depth
    depth = depth.cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
    valid_depth = depth[depth > 0]
    if valid_depth.size == 0:
        return None

    distance = float(np.median(valid_depth))
    offset = choice(list(VIEW_OFFSETS.values()))
    side_cam = process_view(viewpoint_cam, offset["azim"], offset["elev"], distance)
    return render(side_cam, gaussians, pipelineParam, background, stage=stage)["render"].unsqueeze(0)


def scene_reconstruction(
    scene,
    modelParam,
    optimizationParam,
    modelHiddenParam,
    pipelineParam,
    saving_iterations, # save gaussians
    checkpoint_iterations, # save checkpoints
    load_checkpoint,
    debug_from,
    stage,
    tb_writer,
    timer,
):
    timer.start()

    first_iter = 0
    scene.gaussians.training_setup(optimizationParam) # set up optimizer
    if load_checkpoint:
        (model_params, first_iter) = torch.load(scene.model_path + "/chkpnt" + str(load_checkpoint) + ".pth") # first_iter: iteration number
        scene.gaussians.restore(model_params, optimizationParam)

    bg_color = [1, 1, 1] if modelParam.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_smoothness = 0.6

    if stage == "coarse":
        final_iter = optimizationParam.coarse_iterations
    elif stage == "fine":
        final_iter = optimizationParam.iterations # train until

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training Progress ({})".format(stage))
    first_iter += 1

    lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoViews() # redundant
    viewpoint_stack = scene.getTrainViews()

    for iteration in range(first_iter, final_iter + 1):

        # network
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None: # have not read
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipelineParam.convert_SHs_python,
                    pipelineParam.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                    ts,
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam,
                        scene.gaussians,
                        pipelineParam,
                        background,
                        scaling_modifer,
                        stage="stage",
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, modelParam.source_path)
                if do_training and (
                    (iteration < int(optimizationParam.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                print(f"Failed to setup Network GUI: {e}")
                network_gui.conn = None

        # render
        iter_start.record()
        scene.gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            scene.gaussians.oneupSHdegree() # how used?
        if stage == "coarse":
            idx = 0 # always first frame?
        else:
            idx = randint(0, len(viewpoint_stack) - 1) # random
        viewpoint_cams = [viewpoint_stack[idx]] # one random view

        if (iteration - 1) == debug_from:
            pipelineParam.debug = True # ?

        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []

        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        for viewpoint_cam in viewpoint_cams:

            # render
            render_pkg = render(viewpoint_cam, scene.gaussians, pipelineParam, background, stage=stage)
            image, depth, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()

            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0) # visible on any image
        rendered_images = torch.cat(images, 0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images, 0)
        gt_depths = torch.cat(gt_depths, 0)
        masks = torch.cat(masks, 0)

        # loss
        Ll1 = l1_loss(rendered_images, gt_images, masks)

        if (gt_depths != 0).sum() < 10:
            depth_loss = torch.tensor(0.0).cuda() # no depth
        elif scene.mode == "binocular": # inverse loss
            rendered_depths[rendered_depths != 0] = (
                1 / rendered_depths[rendered_depths != 0]
            )
            gt_depths[gt_depths != 0] = 1 / gt_depths[gt_depths != 0]
            depth_loss = l1_loss(rendered_depths, gt_depths, masks)
        elif scene.mode == "monocular":
            rendered_depths_reshape = rendered_depths.reshape(-1, 1)
            gt_depths_reshape = gt_depths.reshape(-1, 1)
            mask_tmp = mask.reshape(-1)
            rendered_depths_reshape, gt_depths_reshape = (
                rendered_depths_reshape[mask_tmp != 0, :],
                gt_depths_reshape[mask_tmp != 0, :],
            )
            depth_loss = 0.001 * (
                1 - pearson_corrcoef(gt_depths_reshape, rendered_depths_reshape)
            )
        else:
            raise ValueError(f"{scene.mode} is not implemented.")

        depth_tvloss = TV_loss(rendered_depths)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = 0.03 * (img_tvloss + depth_tvloss)

        loss = Ll1 + depth_loss + tv_loss

        if stage == "fine" and modelHiddenParam.time_smoothness_weight != 0:
            tv_loss = scene.gaussians.compute_regularization() # time_smoothness_weight, l1_time_planes_weight, plane_tv_weight
            loss += tv_loss
        if optimizationParam.lambda_dssim != 0:
            ssim_loss = ssim(rendered_images, gt_images)
            loss += optimizationParam.lambda_dssim * (1.0 - ssim_loss)
        if optimizationParam.lambda_lpips != 0:
            lpipsloss = lpips_loss(rendered_images, gt_images, lpips_model)
            loss += optimizationParam.lambda_lpips * lpipsloss

        if (
            optimizationParam.sideview_smooth_weight != 0
            and optimizationParam.sideview_reg_interval > 0
            and iteration % optimizationParam.sideview_reg_interval == 0
        ):
            side_render = render_side_view(viewpoint_cams[0], scene.gaussians, pipelineParam, background, stage)
            if side_render is not None:
                loss += optimizationParam.sideview_smooth_weight * TV_loss(side_render)

        if optimizationParam.anisotropy_weight != 0:
            loss += optimizationParam.anisotropy_weight * anisotropy_loss(
                scene.gaussians.get_scaling,
                optimizationParam.anisotropy_ratio_threshold,
                optimizationParam.anisotropy_size_power,
            )

        loss.backward()

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = (
                viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            )
        iter_end.record()

        # record

        psnr_ = psnr(rendered_images, gt_images, masks).mean().double() # not part of loss

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = (1 - ema_smoothness) * loss.item() + ema_smoothness * ema_loss_for_log # Exponential Moving Average
            ema_psnr_for_log = (1 - ema_smoothness) * psnr_ + ema_smoothness * ema_psnr_for_log
            total_point = scene.gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "psnr": f"{psnr_:.{2}f}",
                        "point": f"{total_point}",
                    }
                )
                progress_bar.update(10) # advance 10 steps and print
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            if tb_writer:
                training_report(
                    tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    iter_start.elapsed_time(iter_end),
                    stage,
                )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

            # render process
            if modelParam.render_process: # default false
                if (
                    (iteration < 1000 and iteration % 10 == 1)
                    or (iteration < 3000 and iteration % 50 == 1)
                    or (iteration < 10000 and iteration % 100 == 1)
                    or (iteration < 60000 and iteration % 100 == 1)
                ):
                    render_training_image(
                        scene,
                        video_cams,
                        render,
                        pipelineParam,
                        background,
                        stage,
                        iteration - 1,
                        timer.get_elapsed_time(),
                    )
            timer.start()

            # Densification and pruning
            if iteration < optimizationParam.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                scene.gaussians.max_radii2D[visibility_filter] = torch.max(
                    scene.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                scene.gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter
                )

                if stage == "coarse":
                    opacity_threshold = optimizationParam.opacity_threshold_coarse
                    densify_threshold = optimizationParam.densify_grad_threshold_coarse
                else:
                    opacity_threshold = optimizationParam.opacity_threshold_fine_init + iteration * (
                        optimizationParam.opacity_threshold_fine_after
                        - optimizationParam.opacity_threshold_fine_init
                        ) / (optimizationParam.densify_until_iter)
                    densify_threshold = optimizationParam.densify_grad_threshold_fine_init + iteration * (
                            optimizationParam.densify_grad_threshold_after
                            - optimizationParam.densify_grad_threshold_fine_init
                            ) / (optimizationParam.densify_until_iter)

                if (
                    iteration > optimizationParam.densify_from_iter
                    and iteration % optimizationParam.densification_interval == 0
                ):
                    size_threshold = (
                        optimizationParam.densify_size_threshold
                        if iteration > optimizationParam.opacity_reset_interval
                        else None
                    )
                    scene.gaussians.densify(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if (
                    iteration > optimizationParam.pruning_from_iter
                    and iteration % optimizationParam.pruning_interval == 0
                ):
                    size_threshold = (
                        optimizationParam.prune_size_threshold
                        if iteration > optimizationParam.opacity_reset_interval
                        else None
                    )
                    scene.gaussians.prune(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        optimizationParam.prune_scale_extent_ratio,
                    )

                if iteration % optimizationParam.opacity_reset_interval == 0 or (
                    modelParam.white_background and iteration == optimizationParam.densify_from_iter # ?
                ):
                    print("reset opacity")
                    scene.gaussians.reset_opacity()

            # Optimizer step
            if iteration < final_iter:
                scene.gaussians.optimizer.step()
                scene.gaussians.optimizer.zero_grad(set_to_none=True)

            # Save checkpoint
            if iteration in checkpoint_iterations: # default none
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (scene.gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def training(
    modelParam,
    modelHiddenParam,
    optimizationParam,
    pipelineParam,
    saving_iterations,
    checkpoint_iterations,
    load_checkpoint,
    debug_from,
    extra_mark,
    tb_writer,
):

    gaussians = GaussianModel(modelParam.sh_degree, modelHiddenParam)
    timer = Timer()
    scene = Scene(modelParam, gaussians, load_coarse=modelParam.no_fine) # scene contains Gaussians
    scene_reconstruction(
        scene,
        modelParam,
        optimizationParam,
        modelHiddenParam,
        pipelineParam,
        saving_iterations,
        checkpoint_iterations,
        load_checkpoint,
        debug_from,
        "coarse",
        tb_writer,
        timer,
    )
    if not modelParam.no_fine:
        scene_reconstruction(
            scene,
            modelParam,
            optimizationParam,
            modelHiddenParam,
            pipelineParam,
            saving_iterations,
            checkpoint_iterations,
            load_checkpoint,
            debug_from,
            "fine",
            tb_writer,
            timer,
        )


if __name__ == "__main__":

    torch.cuda.empty_cache()
    set_seed(0)
    set_seed_train(6666)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters (with side-view smoothness regularization)")
    modelParam = ModelParams()
    optimizationParam = OptimizationParams()
    pipelineParam = PipelineParams()
    modelHiddenParam = ModelHiddenParams()

    modelParam.register(parser)
    optimizationParam.register(parser)
    pipelineParam.register(parser)
    modelHiddenParam.register(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--save_iterations",
        nargs="+", # one or more values
        type=int,
        default=[
            2000,
            3000,
            4000,
            5000,
            6000,
            9000,
            10000,
            14000,
            20000,
            30_000,
            45000,
            60000,
        ],
    )
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None) # ?
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)  # save after last iteration

    if args.expname:
        if not args.model_path:
            args.model_path = os.path.join("./output/", args.expname)
        if not args.configs:
            args.configs = os.path.join("./arguments/", args.expname + ".py")
        if not args.source_path:
            args.source_path = os.path.join("./data/", args.expname)

    # configs > cmdline > default
    if args.configs:
        import mmcv
        config = mmcv.Config.fromfile(args.configs) # read and return similar to dict
        args = merge_hparams(args, config)  # overwrite args with config

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port) # ???
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    format_output(args.quiet)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    # save parameters
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    training(
        modelParam.extract(args),
        modelHiddenParam.extract(args),
        optimizationParam.extract(args),
        pipelineParam.extract(args),
        args.save_iterations,
        args.checkpoint_iterations,
        args.load_checkpoint,
        args.debug_from,
        args.extra_mark,
        tb_writer
    )

    # All done
    print("\nTraining complete.")
