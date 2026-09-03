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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def TV_loss(x):
    B, C, H, W = x.shape
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * C * H * W)


def huber_loss(x, beta=0.05):
    # Huber analogue of TV_loss: same gradient-vs-zero penalty, but quadratic below `beta`
    # and linear above it, so real discontinuities (tissue folds, floater edges) aren't
    # over-penalized the way a pure L2 smoothness term would. Drop-in for TV_loss on any
    # single tensor (color or depth).
    # pass the shifted slices directly as (input, target) -- smooth_l1_loss/l1_loss already
    # compute |input - target| internally, so subtracting first and comparing to zero would
    # just be a redundant extra subtraction and allocation
    if beta == 0:
        # Huber degenerates to L1 as beta -> 0; special-cased rather than passed through to
        # smooth_l1_loss, whose quadratic branch divides by beta (would be 0/0 at beta=0)
        return F.l1_loss(x[:, :, 1:, :], x[:, :, :-1, :]) + F.l1_loss(x[:, :, :, 1:], x[:, :, :, :-1])
    return F.smooth_l1_loss(x[:, :, 1:, :], x[:, :, :-1, :], beta=beta) + F.smooth_l1_loss(x[:, :, :, 1:], x[:, :, :, :-1], beta=beta)


def anisotropy_loss(scaling, ratio_power=1.0, size_power=1.0, ratio_threshold=-1):
    # weighted by scaling_max ** size_power (size_power < 1 -> sub-linear growth with size,
    # so small needles are cheap and large ones are not, without dominating other losses).
    # the weight is detached so this loss only pushes on the ratio (shape), never on size itself.
    scaling_max = scaling.max(dim=1).values
    scaling_min = scaling.min(dim=1).values
    ratio = scaling_max / (scaling_min + 1e-8)
    if ratio_threshold == -1:
        # continuous penalty on per-gaussian max/min scale ratio: (ratio - 1) ** ratio_power.
        # ratio_power > 1 makes the penalty grow super-linearly with anisotropy, so it stays
        # negligible near ratio=1 and squashes the far tail hard -- unlike the hinge below,
        # this has nonzero gradient everywhere and shrinks the whole distribution instead
        # of just clipping outliers above a fixed cutoff.
        penalty = (ratio - 1).pow(ratio_power)
    else:
        # squared hinge: zero penalty (and zero gradient) at or below the threshold, so
        # ratios settle uniformly within that range; only the excess above threshold is
        # squashed, growing smoothly (no gradient kink at the threshold like a linear hinge).
        penalty = F.relu(ratio - ratio_threshold).pow(ratio_power)
    return (penalty * scaling_max.detach().pow(size_power)).mean()


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None):
    loss = torch.abs((network_output - gt))
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        elif mask.ndim == 3:
            mask = mask.repeat(network_output.shape[1], 1, 1)
        else:
            raise ValueError('the dimension of mask should be either 3 or 4')

        try:
            loss = loss[mask!=0]
        except Exception as e:
            print("Masking error:", e)
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
    return loss.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
