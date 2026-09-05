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
from arguments import ParamGroup
from utils.general_utils import searchForMaxIteration

from scene.datasets import EndoNeRF_Dataset, SCARED_Dataset, Hamlyn_Dataset
from scene.gaussian_model import GaussianModel


class Scene:

    def __init__(self, modelParam : ParamGroup, gaussians : GaussianModel, load_iteration=None, load_coarse=False):

        self.model_path = modelParam.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.mode = modelParam.mode # mono or binocular
        # view_angle = (azim, elev)

        if load_iteration:
            if load_iteration == -1: # load maximum checkpoint
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        dataset = None

        # render.py registers ModelParams with set_default_none=True, so get_combined_args drops
        # init_pts when the checkpoint's saved cfg_args predates it. Leave it out of the kwargs in
        # that case and let each loader's own default apply, rather than repeating the number here.
        loader_args = {"mode": modelParam.mode}
        init_pts = getattr(modelParam, "init_pts", None)
        if init_pts is not None:
            loader_args["init_pts"] = init_pts

        if os.path.exists(os.path.join(modelParam.source_path, "poses_bounds.npy")) and modelParam.extra_mark == 'endonerf':
            dataset = EndoNeRF_Dataset(modelParam.source_path, **loader_args)
            print("Found poses_bounds.py and extra marks with EndoNeRf")

        elif os.path.exists(os.path.join(modelParam.source_path, "poses_bounds.npy")) and modelParam.extra_mark == 'hamlyn':
            dataset = Hamlyn_Dataset(modelParam.source_path, **loader_args)
            print("Found poses_bounds.py and extra marks with Hamlyn")

        elif os.path.exists(os.path.join(modelParam.source_path, "point_cloud.obj")) or os.path.exists(os.path.join(modelParam.source_path, "left_point_cloud.obj")):
            dataset = SCARED_Dataset(modelParam.source_path, **loader_args)
            print("Found point_cloud.obj, assuming SCARED data!")

        else:
            assert False, "Could not recognize scene type!"

        if dataset is not None:
            scene_info = dataset.get_scene_info()

        self.maxtime = scene_info.maxtime

        if modelParam.extra_mark != 'scared':
            self.cameras_extent = modelParam.camera_extent
        else:
            self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("self.cameras_extent is ", self.cameras_extent)

        print("Loading Training Views")
        self.train_views = scene_info.train_views
        print("Loading Test Views")
        self.test_views = scene_info.test_views
        print("Loading Video Views")
        self.video_views = scene_info.video_views

        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)

        self.gaussians._deform_net.hexplane.set_aabb(xyz_max,xyz_min)

        if self.loaded_iter:
            iteration_str = 'iteration_'+str(self.loaded_iter) if not load_coarse else 'coarse_iteration_'+str(self.loaded_iter)
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           iteration_str,
                                                           "point_cloud.ply"))
            self.gaussians.load_deformation(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    iteration_str,
                                                   )) # deformation
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)



    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def getTrainViews(self, scale=1.0):
        return self.train_views

    def getTestViews(self, scale=1.0):
        return self.test_views

    def getVideoViews(self, scale=1.0):
        return self.video_views
