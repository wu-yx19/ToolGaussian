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
import numpy as np
import math
import open3d as o3d
import torch
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def create_rotation_matrix(azim_deg, elev_deg):

    azim_rad = np.radians(azim_deg)
    axis = np.array([- np.sin(azim_rad), np.cos(azim_rad), 0], dtype=float)
    axis /= np.linalg.norm(axis)
    rotation_matrix = R.from_rotvec(axis * np.radians(elev_deg)).as_matrix()
    return rotation_matrix

def process_view(view, azim, elev, dist):

    view_new = deepcopy(view)
    rotation_matrix = create_rotation_matrix(azim, elev)

    W2C0 = np.zeros((4, 4))
    W2C0[:3,:3] = view.R.transpose()
    W2C0[:3,3] = view.T
    W2C0[3,3] = 1

    C12C0 = np.zeros((4, 4))
    C12C0[:3,:3] = rotation_matrix
    C12C0[:3,3] = (np.eye(3)-rotation_matrix) @ np.array([0, 0, dist]) # C1 in C0
    C12C0[3,3] = 1

    W2C1 = np.linalg.inv(C12C0) @ W2C0
    view_new.R = W2C1[:3,:3].T
    view_new.T = W2C1[:3,3]
    view_new.update_transform()
    return view_new

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def unproject_depth(depth, fx, fy, cx, cy):
    # camera-space Z (z-buffer) depth, per get_pts_cam in scene/datasets.py
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    return np.stack([x, y, depth], axis=-1)  # HxWx3, camera space

def warp_view(image, depth, src_R, src_T, tgt_R, tgt_T, fovx, fovy, width, height):
    # forward-warps image/depth (captured by the src camera) into the tgt camera's view via
    # a z-buffered point splat. src and tgt are assumed to share intrinsics (fovx, fovy, width, height),
    # which holds for any pair of views produced by process_view (it only ever changes R/T).
    fx, fy = fov2focal(fovx, width), fov2focal(fovy, height)
    cx, cy = width / 2, height / 2
    src_R, src_T, tgt_R, tgt_T = np.asarray(src_R), np.asarray(src_T), np.asarray(tgt_R), np.asarray(tgt_T)

    valid = depth > 0
    pts_src = unproject_depth(depth, fx, fy, cx, cy)[valid]
    colors = image[valid]

    # Convention (getWorld2View2 above): P_cam_col = R.T @ P_world_col + T, so for row-vector
    # points: P_world_row = (P_cam_row - T_row) @ R.T, and P_cam_row = P_world_row @ R + T_row.
    pts_world = (pts_src - src_T) @ src_R.T
    pts_tgt = pts_world @ tgt_R + tgt_T

    in_front = pts_tgt[:, 2] > 0
    pts_tgt = pts_tgt[in_front]
    colors = colors[in_front]

    u_tgt = fx * pts_tgt[:, 0] / pts_tgt[:, 2] + cx
    v_tgt = fy * pts_tgt[:, 1] / pts_tgt[:, 2] + cy
    u_tgt_i = np.round(u_tgt).astype(np.int64)
    v_tgt_i = np.round(v_tgt).astype(np.int64)

    in_bounds = (u_tgt_i >= 0) & (u_tgt_i < width) & (v_tgt_i >= 0) & (v_tgt_i < height)
    u_tgt_i, v_tgt_i = u_tgt_i[in_bounds], v_tgt_i[in_bounds]
    z_tgt = pts_tgt[in_bounds, 2]
    colors = colors[in_bounds]

    warped = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    mask = np.zeros((height, width), dtype=bool)

    # fancy-index assignment keeps the LAST write for duplicate (v, u) indices, so sorting
    # far-to-near and assigning once gives z-buffer "nearest wins" semantics without a python loop
    order = np.argsort(-z_tgt)
    v_sorted, u_sorted = v_tgt_i[order], u_tgt_i[order]
    warped[v_sorted, u_sorted] = colors[order]
    mask[v_sorted, u_sorted] = True

    return warped, mask

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale # move camera center around
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def reconstruct_point_cloud(images, masks, depths, camera_parameters, output_frame_folder):
    import copy
    import cv2

    os.makedirs(output_frame_folder, exist_ok=True)
    frames = np.arange(len(images))
    # frames = [0]
    focal_x, focal_y, width, height = camera_parameters
    for i_frame in frames:
        rgb_tensor = images[i_frame]
        rgb_np = (
            rgb_tensor.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .contiguous()
            .to("cpu")
            .numpy()
        )
        depth_np = depths[i_frame].cpu().numpy()
        depth_np = depth_np.squeeze(0)
        mask = masks[i_frame]
        mask = mask.squeeze(0).cpu().numpy()

        rgb_new = copy.deepcopy(rgb_np)

        depth_smoother = (128, 64, 64)
        depth_np = cv2.bilateralFilter(
            depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2]
        )

        close_depth = np.percentile(depth_np[depth_np != 0], 5)
        inf_depth = np.percentile(depth_np, 95)
        depth_np = np.clip(depth_np, close_depth, inf_depth)

        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_im, depth_im, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                width, height, focal_x, focal_y, width / 2, height / 2 # width/2?
            ),
            project_valid_depth_only=True,
        )
        o3d.io.write_point_cloud(
            os.path.join(output_frame_folder, "frame_{}.ply".format(i_frame)), pcd
        )

@torch.no_grad()
def render_training_image(scene, viewpoints, render_func, pipe, background, stage, iteration, time_now):
    def render(gaussians, viewpoint, path, scaling):
        # scaling_copy = gaussians._scaling
        render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage)
        label1 = f"stage:{stage},iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为 (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)
        image_np = np.concatenate((image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  # 转换为8位图像
        # 创建PIL图像对象的副本以绘制标签
        draw1 = ImageDraw.Draw(image_with_labels)

        # 选择字体和字体大小
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)  # 请将路径替换为您选择的字体文件路径

        # 选择文本颜色
        text_color = (255, 0, 0)  # 白色

        # 选择标签的位置（左上角坐标）
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # 右上角坐标

        # 在图像上添加标签
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)

        image_with_labels.save(path)
    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # image:3,800,800

    # point_save_path = os.path.join(point_cloud_path,f"{iteration}.jpg")
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.jpg")
        render(scene.gaussians,viewpoints[idx],image_save_path,scaling = 1)
    # render(gaussians,point_save_path,scaling = 0.1)
    # 保存带有标签的图像

    pc_mask = scene.gaussians.get_opacity
    pc_mask = pc_mask > 0.1
