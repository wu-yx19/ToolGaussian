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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from typing import NamedTuple

# class ViewInfo(NamedTuple):
#     uid: int
#     R: np.array
#     T: np.array
#     FovY: np.array
#     FovX: np.array
#     image: np.array
#     depth: np.array
#     image_path: str
#     image_name: str
#     width: int
#     height: int
#     time : float
#     mask: np.array
#     Zfar: float
#     Znear: float

class View(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, mask, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 data_device = "cuda", time = 0, Znear=None, Zfar=None
                 ):
        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.mask = mask
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)
        self.original_depth = depth
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        if Zfar is not None and Znear is not None:
            self.zfar = Zfar
            self.znear = Znear
        else:
            # ENDONERF
            self.zfar = 120.0
            self.znear = 0.01
            # SCARED
            # self.zfar = 300
            # self.znear= 0.03

        self.trans = trans
        self.scale = scale
        self.update_transform()

        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]

    def update_transform(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
