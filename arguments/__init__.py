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
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, name: str = ""):
        self._group_name = name

    def register(self, parser: ArgumentParser, set_default_none=False):
        # register parameters to parser
        # set_default_none: whether to repalce stored default value with none
        group = parser.add_argument_group(self._group_name)
        for key, value in vars(self).items():
            if key == "_group_name":
                continue
            shorthand = False
            if key.startswith("_"): # allow shorthand
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not set_default_none else None
            if shorthand:
                if isinstance(value, bool):
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if isinstance(value, bool):
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument(
                        "--" + key, default=value, type=t
                    )  # add to parser

    def extract(self, args) -> GroupParams:
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

# paramter meaning?
#
class ModelParams(ParamGroup): # model loading params
    def __init__(self):
        super().__init__("ModelParams")
        self.sh_degree = 3 #
        self._source_path = "" # original data
        self._model_path = "" # saved model
        # self._images = "images"
        # self._resolution = -1
        self._white_background = False #
        # self.data_device = "cuda"
        # self.eval = True
        self.render_process = False #
        self.extra_mark = None #
        self.camera_extent = None #
        self.mode = "binocular" #
        self.no_fine = False #
        # self.init_pts = 200_000

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self):
        super().__init__("PipelineParams")
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class ModelHiddenParams(ParamGroup): # deformation
    def __init__(self):
        super().__init__("ModelHiddenParams")
        self.net_width = 64 #
        self.defor_depth = 1 #
        self.bounds = 1.6 #
        self.plane_tv_weight = 0.0001 #
        self.time_smoothness_weight = 0.01 #
        self.l1_time_planes = 0.0001 #
        self.kplanes_config = {
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 32,
            "resolution": [64, 64, 64, 25],
        } #
        self.multires = [1, 2, 4, 8] #
        self.no_dx = False #
        self.no_grid = False #
        self.no_ds = False #
        self.no_dr = False #
        self.no_do = False #


class OptimizationParams(ParamGroup):
    def __init__(self):
        super().__init__("OptimizationParams")
        # self.dataloader = False
        self.iterations = 30_000 #
        self.coarse_iterations = 3000 #
        self.position_lr_init = 0.00016 #
        self.position_lr_final = 0.0000016 #
        self.position_lr_delay_mult = 0.01 #
        self.position_lr_max_steps = 20_000 #
        self.deformation_lr_init = 0.00016 #
        self.deformation_lr_final = 0.000016 #
        self.deformation_lr_delay_mult = 0.01 #
        self.grid_lr_init = 0.0016 #
        self.grid_lr_final = 0.00016 #

        self.feature_lr = 0.0025 # sh
        self.opacity_lr = 0.05 #
        self.scaling_lr = 0.005 #
        self.rotation_lr = 0.001 #
        self.percent_dense = 0.01 #
        self.lambda_dssim = 0 #
        self.lambda_lpips = 0 #
        self.weight_constraint_init = 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = -1 # -1 disables periodic opacity reset entirely
        self.opacity_reset_value = 0.01 # opacity ceiling applied to every point on reset
        self.densification_interval = 100 #
        self.densify_from_iter = 500 #
        self.densify_until_iter = 15_000 #
        self.densify_grad_threshold_coarse = 0.0002 #
        self.densify_grad_threshold_fine_init = 0.0002 #
        self.densify_grad_threshold_after = 0.0002 #
        self.pruning_from_iter = 500 #
        self.pruning_interval = 100 #
        self.prune_size_threshold = 40 # max_radii2D screen-size cutoff during prune
        self.prune_scale_extent_ratio = 0.1 # scale.max() > ratio * extent -> pruned as floater
        self.size_prune_grace_period = 500 # iterations after each opacity reset before size-based pruning re-arms
        self.max_prune_fraction = 0.5 # safety cap: prune() falls back to opacity-only if size criteria would remove more than this fraction
        self.sideview_smooth_weight = 0.03 # TV-loss weight on a rendered synthetic side view (train-sideview.py)
        self.sideview_reg_interval = -1 # render+regularize a side view every N iterations; -1 disables it
        self.sideview_elev = 20 # elevation offset (degrees) used to synthesize the side view (train-sideview.py)
        self.sideview_azims = [0, 90, 180, 270] # azimuth offsets (degrees) sampled from for the side view; -1 = sample uniformly from [0, 360) instead
        self.anisotropy_weight = 0 # penalizes per-gaussian scale max/min ratio beyond anisotropy_ratio_threshold; 0 disables it (train-sideview.py)
        self.anisotropy_ratio_threshold = 5.0 # max/min scale ratio allowed before penalty kicks in
        self.anisotropy_size_power = 0.5 # penalty is weighted by scaling_max ** this; <1 grows sub-linearly with gaussian size
        self.opacity_threshold_coarse = 0.005 #
        self.opacity_threshold_fine_init = 0.005 #
        self.opacity_threshold_fine_after = 0.005 #

##
def get_combined_args(args_cmdline):
    # overwrite parameters with file
    cfgfile_string = "Namespace()"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict) # dict with keys -> object with attributes


def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)
    return args
