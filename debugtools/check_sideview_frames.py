"""Replay sideview.py's arg pipeline for a trained run and print the frame-selection inputs."""

import sys
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, ModelHiddenParams, SideviewParams, get_combined_args, merge_hparams
from utils.general_utils import resolve_expname_paths

EXPNAME = sys.argv[1] if len(sys.argv) > 1 else "scared/d2k1-depthreg-aniso1e5-depth002"

# exactly what sbatch/warpback_single.sh passes
argv = ["--expname", EXPNAME, "--save_depth", "--save_meta", "--sideview_on_test",
        "--elev", "5", "10", "15", "20", "30", "45"]

parser = ArgumentParser()
modelParam, pipelineParam = ModelParams(), PipelineParams()
modelHiddenParam, sideviewParam = ModelHiddenParams(), SideviewParams()
modelParam.register(parser, set_default_none=True)
pipelineParam.register(parser, set_default_none=True)
modelHiddenParam.register(parser, set_default_none=True)
sideviewParam.register(parser)
parser.add_argument("--configs", type=str)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--expname", type=str, default="")

args = parser.parse_args(argv)
print("after parse_args      :", args.frame_stride, args.frame_idxs, args.sideview_on_test)

configs_auto_derived = bool(args.expname and not args.configs)
args = resolve_expname_paths(args)
args = get_combined_args(args)
print("after get_combined    :", args.frame_stride, args.frame_idxs, args.sideview_on_test)

import os.path
if args.configs and (not configs_auto_derived or os.path.isfile(args.configs)):
    import mmcv
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    print(f"after merge {args.configs}: {args.frame_stride} {args.frame_idxs} {args.sideview_on_test}")

sv = sideviewParam.extract(args)
print("after extract         :", sv.frame_stride, sv.frame_idxs, sv.sideview_on_test)
print("-> branch taken       :",
      "video" if (sv.frame_stride or sv.frame_idxs) else ("test" if sv.sideview_on_test else "none"))

# build the Scene exactly as sideview.py does and report what the test split actually contains
import torch
from scene import Scene
from scene.gaussian_renderer import GaussianModel

mp = modelParam.extract(args)
mhp = modelHiddenParam.extract(args)
print("source_path           :", mp.source_path)
print("mode                  :", mp.mode)
with torch.no_grad():
    scene = Scene(mp, GaussianModel(mp.sh_degree, mhp),
                  load_iteration=args.iteration, load_coarse=mp.no_fine)
    test_views = scene.getTestViews()
    print("len(test_views)       :", len(test_views))
    print("test uids             :", [v.uid for v in test_views])
    print("len(video_views)      :", len(scene.getVideoViews()))
