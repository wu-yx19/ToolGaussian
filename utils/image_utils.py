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
import glob
import imageio
import cv2
from PIL import Image
import torchvision.transforms.functional as tf


def to8b(x):
    return (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)  # tensor to uint8

def add_title(image, title):
    # returns a labeled copy; leaves the input array untouched so callers can still
    # save/reuse the original unlabeled image (e.g. individual per-view renders)
    image = image.copy()
    cv2.putText(image, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def masked_local_variance(gray, mask, ksize=15):
    # windowed intensity variance via normalized convolution: unbiased even where `mask`
    # has holes, as long as a window has some valid coverage. gray: float array (H, W).
    # mask: 0/1 float array (H, W). Returns (variance, coverage) where coverage is the
    # fraction of each window that was valid -- callers should discard low-coverage windows.
    gray = gray.astype(np.float32)
    mask = mask.astype(np.float32)
    masked = gray * mask

    weight = cv2.boxFilter(mask, -1, (ksize, ksize), normalize=False)
    safe_weight = np.maximum(weight, 1e-6)
    mean = cv2.boxFilter(masked, -1, (ksize, ksize), normalize=False) / safe_weight
    mean_sq = cv2.boxFilter(masked * gray, -1, (ksize, ksize), normalize=False) / safe_weight
    variance = np.clip(mean_sq - mean ** 2, 0, None)
    coverage = weight / (ksize * ksize)
    return variance, coverage

def masked_ssim_map(gray1, gray2, mask, window=11):
    # windowed SSIM between two grayscale images via normalized convolution (same pattern as
    # masked_local_variance, extended to a joint covariance term): unbiased even where `mask`
    # has holes. gray1/gray2: float arrays (H, W), 0-255 scale. mask: 0/1 float array (H, W).
    # Returns (ssim_map, coverage).
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    gray1, gray2, mask = gray1.astype(np.float32), gray2.astype(np.float32), mask.astype(np.float32)

    def box(x):
        return cv2.boxFilter(x, -1, (window, window), normalize=False)

    weight = box(mask)
    safe_weight = np.maximum(weight, 1e-6)
    mu1 = box(gray1 * mask) / safe_weight
    mu2 = box(gray2 * mask) / safe_weight
    sigma1_sq = box(gray1 * gray1 * mask) / safe_weight - mu1 ** 2
    sigma2_sq = box(gray2 * gray2 * mask) / safe_weight - mu2 ** 2
    sigma12 = box(gray1 * gray2 * mask) / safe_weight - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    coverage = weight / (window * window)
    return ssim_map, coverage

def masked_psnr(img1, img2, valid):
    # img1/img2: uint8 or float arrays, same shape (HxW or HxWxC). valid: bool array (H, W),
    # broadcast across channels if present. 8-bit (0-255) scale.
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    valid_full = np.broadcast_to(valid[..., None], diff.shape) if diff.ndim == 3 else valid
    mse = np.mean(diff[valid_full] ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))

def masked_gradient_energy(gray, mask, ksize=3):
    # Sobel gradient magnitude; only trustworthy where the full ksize x ksize neighborhood is
    # valid, so `valid` erodes `mask` by a matching kernel instead of using normalized
    # convolution (the kernel is small enough that this is simpler and exact rather than
    # approximate). Larger ksize smooths the derivative -- less noise-sensitive, but also
    # blurs away fine texture, so it measures richness at a coarser scale.
    # gray: float array (H, W). mask: 0/1 array (H, W). Returns (magnitude, valid).
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    valid = cv2.erode(mask.astype(np.uint8), np.ones((ksize, ksize), np.uint8)).astype(bool)
    return magnitude, valid

def compute_texture_density(gray, mask, window=15, coverage_thresh=0.5, grad_ksize=3):
    # runs both texture metrics for one grayscale image + validity mask.
    # gray: float array (H, W). mask: bool array (H, W), True = valid.
    # Returns (variance, valid_var, gradient, valid_grad).
    variance, coverage = masked_local_variance(gray, mask.astype(np.float32), window)
    valid_var = mask & (coverage >= coverage_thresh)

    gradient, grad_core = masked_gradient_energy(gray, mask.astype(np.uint8), grad_ksize)
    valid_grad = mask & grad_core

    return variance, valid_var, gradient, valid_grad

def load_gray(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

def texture_heatmap(values, valid, vmax=None):
    # values/valid -> a viridis heatmap PNG (uint8 BGR), invalid pixels forced to black.
    # vmax fixes the color scale; pass the same vmax across images to keep heatmaps
    # visually comparable (otherwise each image's own max would rescale independently).
    if vmax is None:
        vmax = values[valid].max() if valid.any() else 1.0
    vis = np.clip(values / max(vmax, 1e-6), 0, 1)
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    vis[~valid] = 0
    return vis

def sideview_view_elevs(elev):
    # mirrors sideview.py's get_view_offsets: same canonical view names/order, and the fact
    # that "central" always renders at elev_deg=0 (and so lands in elev0/) regardless of
    # the requested elev -- every other view renders (and lands) at the requested elev
    return [("central", 0.0), ("left", elev), ("right", elev), ("up", elev), ("down", elev)]

def locate_sideview_render(sideview_dir, frame_idx, view_name, view_elev):
    # sideview_dir: the 'ours_<iteration>' dir (parent of elev<E> folders). Returns
    # (render_path, view_elev_dir), or (None, None) if that view wasn't rendered.
    view_elev_dir = os.path.join(sideview_dir, f"elev{view_elev:.0f}")
    pattern = os.path.join(view_elev_dir, "renders", f"render_frame{frame_idx}_{view_name}_*.png")
    matches = sorted(glob.glob(pattern))
    return (matches[0], view_elev_dir) if matches else (None, None)

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
