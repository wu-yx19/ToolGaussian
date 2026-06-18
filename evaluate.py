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

from pathlib import Path
import os
import torch

import json
from tqdm import tqdm

from utils.loss_utils import ssim
from utils.image_utils import to8b, readImages
from utils.eval_utils import psnr, rmse, flip, cal_lpips, LPIPS

from argparse import ArgumentParser

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    print("")
    lpips_obj = LPIPS()

    with torch.no_grad():
        for scene_dir in model_paths:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                # prepare directories
                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                depth_dir = method_dir / "depth"
                gt_depth_dir = method_dir / "gt_depth"
                masks_dir = method_dir / "masks" # 1 valid

                renders, gts, depths, gt_depths, masks, image_names = readImages(renders_dir, gt_dir, depth_dir, gt_depth_dir, masks_dir)

                ssims = []
                psnrs = []
                psnrs_star = []
                lpipss = []
                rmses = []

                render_wmask = [] # with mask
                gt_wmask = []

                # calculate metrics
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render, gt, depth, gt_depth, mask = renders[idx], gts[idx], depths[idx], gt_depths[idx], masks[idx]

                    psnrs_star.append(psnr(render, gt, mask)) # only evaluate valid region

                    render = render * mask
                    gt = gt * mask
                    render_wmask.append(render)
                    gt_wmask.append(gt)
                    psnrs.append(psnr(render, gt))
                    ssims.append(ssim(render, gt))
                    lpipss.append(cal_lpips(lpips_obj, render, gt))

                    if (gt_depth!=0).sum() < 10:
                        continue

                    tmp_mask = gt_depth != 0
                    depth_mask = torch.logical_and(tmp_mask, mask)
                    depth = depth * depth_mask
                    gt_depth = gt_depth * depth_mask
                    rmses.append(rmse(depth, gt_depth))

                flipped_metrics = flip([to8b(e) for e in render_wmask], [to8b(g) for g in gt_wmask], interval=10) # Fidelity Loss Index Perceptual

                metrics = {
                    "SSIM": ssims,
                    "PSNR": psnrs,
                    "PSNR*": psnrs_star,
                    "LPIPS": lpipss,
                    "FLIP": flipped_metrics,
                    "RMSE": rmses,
                }

                metrics_t = {k: torch.as_tensor(v) for k, v in metrics.items()}
                metrics_mean = {k: v.mean().item() for k, v in metrics_t.items()}

                for name, value in metrics_mean.items():
                    print(f"Scene: {scene_dir}  {name:<6}: {value:12.7f}")

                full_dict[scene_dir][method].update(metrics_mean)

                per_view_metrics = {
                    name: {
                        image_name: metric
                        for metric, image_name in zip(tensor.tolist(), image_names)
                    }
                    for name, tensor in metrics_t.items()
                    if name != "FLIP"
                }

                per_view_dict[scene_dir][method].update(per_view_metrics)

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)

    print("\nEvaluation complete")