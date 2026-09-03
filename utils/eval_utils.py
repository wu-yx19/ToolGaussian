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

import torch
import os
import numpy as np
from typing import List
import re
import subprocess
import tempfile
from PIL import Image
import lpips
from utils.image_utils import array2tensor, tensor2array


# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    """
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """
    def __init__(self, device="cuda"):
        self.model = lpips.LPIPS(net='alex').to(device)

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)

def cal_lpips(lpips_obj: LPIPS, a, b, device="cuda", batch=2):
    """Compute lpips.
    a, b: [batch, H, W, 3]"""
    if not torch.is_tensor(a):
        a = array2tensor(a, device)
    if not torch.is_tensor(b):
        b = array2tensor(b, device)

    lpips_all = []
    for a_split, b_split in zip(a.split(split_size=batch, dim=0), b.split(split_size=batch, dim=0)): # avoid oom
        out = lpips_obj(a_split, b_split)
        lpips_all.append(out)
    lpips_all = torch.stack(lpips_all)
    lpips_mean = lpips_all.mean()
    return lpips_mean

@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is None:
        mse_mask = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        if mask.shape[1] == 3:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10))
        else:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10)*3.0)

    return 20 * torch.log10(1.0 / torch.sqrt(mse_mask))

def write_png(path, data):
    """Writes an PNG image to some path.

    Args:
        path (str): Path to save the PNG.
        data (np.array): HWC image.

    Returns:
        (void): Writes to path.
    """
    Image.fromarray(data).save(path)

def flip(pred_frames: List[np.ndarray], gt_frames: List[np.ndarray], interval: int = 10) -> float:

    def extract_from_result(text: str, prompt: str):
        m = re.search(prompt, text)
        return float(m.group(1))

    all_results = []

    pred_frames = [np.transpose(e.squeeze(0), (1, 2, 0)) for e in pred_frames]
    gt_frames = [np.transpose(e.squeeze(0), (1, 2, 0)) for e in gt_frames]

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_fname = os.path.join(tmpdir, "pred.png")
        gt_fname = os.path.join(tmpdir, "gt.png")
        for i in range(len(pred_frames)):
            write_png(pred_fname, pred_frames[i])
            write_png(gt_fname, gt_frames[i])
            result = subprocess.check_output(
                ['python', 'utils/flip/flip.py', '--reference', gt_fname, '--test', pred_fname]
            ).decode()
            all_results.append(extract_from_result(result, r'Mean: (\d+\.\d+)'))
    return sum(all_results) / len(all_results)

def rmse(a, b, mask=None):
    """Compute rmse.
    """
    if torch.is_tensor(a):
        a = tensor2array(a)
    if torch.is_tensor(b):
        b = tensor2array(b)
    if torch.is_tensor(mask):
        mask = tensor2array(mask)

    if mask is None:
        rmse = (((a - b)**2).sum() / (a.shape[-1]*a.shape[-2]))**0.5
    else:
        if len(mask.shape) == len(a.shape) - 1:
            mask = mask[..., None]
        mask_sum = np.sum(mask) + 1e-10
        rmse = (((a - b)**2 * mask).sum() / (mask_sum))**0.5

    return rmse
