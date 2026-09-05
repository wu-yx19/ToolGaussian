#
# check_accum_alpha.py
#
# Smoke test / sanity check for the rasterizer's "alpha" output (accumulated
# per-pixel opacity, i.e. 1 - final transmittance). Renders a single view from
# a trained model and reports basic stats + a visualization, so a rebuilt
# diff_gaussian_rasterization extension can be verified before trusting it in
# a real training/render/evaluation run.
#
# Usage:
#   python debugtools/check_accum_alpha.py --expname endonerf/pulling
#

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless-safe: no X display needed
import matplotlib.pyplot as plt

from scene.gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import resolve_expname_paths
from utils.image_utils import to8b
from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args, merge_hparams


def main():
    parser = ArgumentParser(description="Sanity-check the rasterizer's alpha output")
    modelParam = ModelParams()
    pipelineParam = PipelineParams()
    modelHiddenParam = ModelHiddenParams()

    modelParam.register(parser, set_default_none=True)
    pipelineParam.register(parser)
    modelHiddenParam.register(parser)

    parser.add_argument("--configs", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--hole_threshold", type=float, default=0.5, help="alpha below this counts as a 'hole' pixel")
    parser.add_argument("--out_dir", type=str, default="test")

    args = parser.parse_args(sys.argv[1:])
    args = resolve_expname_paths(args)
    args = get_combined_args(args)
    if args.configs:
        import mmcv
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    torch.cuda.set_device(torch.device("cuda:0"))

    modelParam = modelParam.extract(args)
    modelHiddenParam = modelHiddenParam.extract(args)
    pipelineParam = pipelineParam.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(modelParam.sh_degree, modelHiddenParam)
        scene = Scene(modelParam, gaussians, load_iteration=args.iteration, load_coarse=modelParam.no_fine)

        bg_color = [1, 1, 1] if modelParam.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        view = scene.getVideoViews()[0]
        stage = "coarse" if modelParam.no_fine else "fine"
        render_pkg = render(view, gaussians, pipelineParam, background, stage=stage)

        assert "alpha" in render_pkg, "render() did not return an 'alpha' key -- rebuild didn't take effect?"

        color = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"]

        print(f"render_pkg keys: {list(render_pkg.keys())}")
        print(f"color: shape={tuple(color.shape)} min={color.min().item():.4f} max={color.max().item():.4f} "
              f"nan={torch.isnan(color).any().item()}")
        print(f"depth: shape={tuple(depth.shape)} min={depth.min().item():.4f} max={depth.max().item():.4f} "
              f"nan={torch.isnan(depth).any().item()}")
        print(f"alpha: shape={tuple(alpha.shape)} dtype={alpha.dtype} "
              f"min={alpha.min().item():.4f} max={alpha.max().item():.4f} mean={alpha.mean().item():.4f} "
              f"nan={torch.isnan(alpha).any().item()}")

        if alpha.shape != depth.shape:
            print(f"WARNING: alpha shape {tuple(alpha.shape)} != depth shape {tuple(depth.shape)}")
        if alpha.min().item() < -1e-4 or alpha.max().item() > 1 + 1e-4:
            print("WARNING: alpha out of expected [0, 1] range")

        hole_frac = (alpha < args.hole_threshold).float().mean().item()
        print(f"hole fraction (alpha < {args.hole_threshold}): {100 * hole_frac:.2f}%")

        os.makedirs(args.out_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        color_np = np.ascontiguousarray(to8b(color.cpu()).transpose(1, 2, 0))
        axes[0].imshow(color_np)
        axes[0].set_title("color")
        axes[0].axis("off")

        im1 = axes[1].imshow(depth.cpu().squeeze().numpy(), cmap="turbo")
        axes[1].set_title("depth")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(alpha.cpu().squeeze().numpy(), cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title(f"alpha (hole={100 * hole_frac:.1f}%)")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        out_path = os.path.join(args.out_dir, "check_accum_alpha.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
