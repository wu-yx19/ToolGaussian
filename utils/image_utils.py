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

import numpy as np
import torch
import torchvision
from tqdm import tqdm
import os
import imageio
import cv2
from PIL import Image
import torchvision.transforms.functional as tf


def to8b(x):
    return (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)  # tensor to uint8

def save_with_title(image, title, path):
    cv2.putText(image, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, image)

def concat_with_title(images, title, path):
    # images: list of HxWx3 uint8 BGR arrays, same size -> one row, with a title bar on top
    row = np.concatenate(images, axis=1)
    bar_h = 40
    title_bar = np.zeros((bar_h, row.shape[1], 3), dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = max((row.shape[1] - text_w) // 2, 10)
    cv2.putText(title_bar, title, (text_x, (bar_h + text_h) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    figure = np.concatenate([title_bar, row], axis=0)
    cv2.imwrite(path, figure)

def write_images(name : str, images, path):
    count = 0
    print("writing {} images.".format(name))
    if len(images) != 0:
        for image in tqdm(images):
            match name:
                case "mask":
                    image = image.float()
                    torchvision.utils.save_image(image, os.path.join(path, "{0:05d}".format(count) + ".png"))
                case "rendered depth":
                    image = np.clip(image.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
                    cv2.imwrite(os.path.join(path, "{0:05d}".format(count) + ".png"), image)
                case "ground truth depth":
                    image = image.cpu().squeeze().numpy().astype(np.uint8)
                    cv2.imwrite(os.path.join(path, "{0:05d}".format(count) + ".png"), image)
                case _:
                    torchvision.utils.save_image(image, os.path.join(path, "{0:05d}".format(count) + ".png"))
            count += 1

def write_video(images, path):
    render_array = torch.stack(images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array * 255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(
        path,
        render_array,
        fps=30,
        quality=8,
    )

def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)

def tensor2array(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor

def readImages(renders_dir, gt_dir, depth_dir, gtdepth_dir, masks_dir):
    renders = [] # rendered image dir
    gts = []
    image_names = []
    depths = []
    gt_depths = []
    masks = []

    for fname in os.listdir(renders_dir):
        render = np.array(Image.open(renders_dir / fname))
        gt = np.array(Image.open(gt_dir / fname))
        depth = np.array(Image.open(depth_dir / fname))
        gt_depth = np.array(Image.open(gtdepth_dir / fname))
        mask = np.array(Image.open(masks_dir / fname))

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        depths.append(torch.from_numpy(depth).unsqueeze(0).unsqueeze(1)[:, :, :, :].cuda())
        gt_depths.append(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(1)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0)[:, 0:1, :, :].cuda())

        image_names.append(fname)
    return renders, gts, depths, gt_depths, masks, image_names
