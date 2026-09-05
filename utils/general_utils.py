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
import re
import torch
import sys
from datetime import datetime
import numpy as np
import random

from errno import EEXIST
from os import makedirs, path

def format_output(silent, log_path=None):
    # log_path: if given, mirror everything written to stdout/stderr into this
    # file too (in addition to the terminal), regardless of `silent`
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file = None
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = open(log_path, "a")

    class F:
        def __init__(self, silent, terminal, log_file):
            self.silent = silent
            self.terminal = terminal
            self.log_file = log_file

        def write(self, x):
            stamped = x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))) if x.endswith("\n") else x
            if not self.silent:
                self.terminal.write(stamped)
            if self.log_file:
                self.log_file.write(stamped)

        def flush(self):
            self.terminal.flush()
            if self.log_file:
                self.log_file.flush()

    sys.stdout = F(silent, old_stdout, log_file)
    sys.stderr = F(False, old_stderr, log_file)

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_seed_train(seed: int = 0):
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def check_tensor(name, t):
    if not torch.isfinite(t).all():
        print(f"{name} has NaN or Inf!")
        return False
    return True

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def mkdir_p(folder_path): # create folder and parents
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

# scared configs are named d<dataset>k<keyframe>, but its data is laid out dataset_<n>/keyframe_<n>
_SCARED_SCENE = re.compile(r"^d(\d+)k(\d+)(?:_mono)?$")

def expname_to_source_path(expname, data_root="./data/"):
    # 'scared/d1k1' -> './data/scared/dataset_1/keyframe_1'; otherwise './data/<expname>'
    dataset, _, scene = expname.partition("/")
    if dataset == "scared":
        match = _SCARED_SCENE.match(scene)
        if match:
            return os.path.join(
                data_root, "scared", f"dataset_{match.group(1)}", f"keyframe_{match.group(2)}"
            )
    return os.path.join(data_root, expname)

def resolve_expname_paths(args, infer_source_path=False):
    # only fills what the caller left unset. infer_source_path is opt-in: render.py and sideview.py
    # take source_path from the checkpoint's saved cfg_args instead
    if not args.expname:
        return args

    if not args.model_path:
        args.model_path = os.path.join("./output/", args.expname)
    if not args.configs:
        args.configs = os.path.join("./arguments/", args.expname + ".py")
    if infer_source_path and not getattr(args, "source_path", None):
        args.source_path = expname_to_source_path(args.expname)

    return args

def training_report(
    tb_writer,
    iteration,
    Ll1,
    depth_loss,
    tv_loss,
    deform_reg_loss,
    loss,
    psnr,
    total_points,
    elapsed,
    stage,
):
    tb_writer.add_scalar(
        f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration
    )
    tb_writer.add_scalar(
        f"{stage}/train_loss_patches/depth_loss", depth_loss.item(), iteration
    )
    tb_writer.add_scalar(
        f"{stage}/train_loss_patches/tv_loss", tv_loss.item(), iteration
    )
    tb_writer.add_scalar(
        f"{stage}/train_loss_patches/deform_reg_loss", deform_reg_loss.item(), iteration
    )
    tb_writer.add_scalar(
        f"{stage}/train_loss_patches/total_loss", loss.item(), iteration
    )
    tb_writer.add_scalar(f"{stage}/train_psnr", psnr.item(), iteration)
    tb_writer.add_scalar(f"{stage}/total_points", total_points, iteration)
    tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)


