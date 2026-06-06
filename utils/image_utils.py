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
import tqdm
import os
import imageio

def to8b(x):
    return (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)  # tensor to uint8

def write_images(name : str, images, path):
    count = 0
    print("writing {} images.".format(name))
    if len(images) != 0:
        for image in tqdm(images):

            match name:
                case "mask":
                    image = image.float()
                case "rendered depth":
                    image = np.clip(image.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
                case "ground truth depth":
                    image = image.cpu().squeeze().numpy().astype(np.uint8)

            torchvision.utils.save_image(
                image, os.path.join(path, "{0:05d}".format(count) + ".png")
            )
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
