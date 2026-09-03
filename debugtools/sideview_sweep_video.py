#
# sideview_sweep_video.py
#
# Demo video sweeping the camera continuously from elev=+45 to elev=-45 at azim=0 (pass
# --elev_min/--elev_max to reverse or change the range). Since +45<-->-45 at a fixed azim is
# just a signed rotation around one axis (see
# utils/graphics_utils.py:create_rotation_matrix), this alone reproduces the "left"/"right"
# offset views sideview.py renders separately (azim=0 vs azim=180) as a single continuous
# sweep -- no need to stitch two directions together.
#
# Two modes:
#   --frame_idx given: freeze that one frame's reconstruction, sweep elev across --num_frames
#     output frames (the "poor -> good -> poor" quality demo).
#   --frame_idx omitted (default): sweep across the WHOLE video sequence instead -- frame 0
#     gets elev_min, the last frame gets elev_max, every frame in between gets its own evenly
#     spaced elev, so the video plays through the whole reconstruction while the camera drifts
#     from one side to the other.
#
# Saves both the assembled mp4 and every individual rendered frame (as PNGs, in a sibling
# "renders" folder) so frames can be inspected or reused without re-decoding the video.
#
# Usage:
#   python debugtools/sideview_sweep_video.py --expname endonerf/cutting-depthreg-aniso1e5-depth002
#   python debugtools/sideview_sweep_video.py --expname endonerf/cutting-depthreg-aniso1e5-depth002 --frame_idx 40 --num_frames 120
#

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root, for `from scene...`/`from utils...`/`from arguments...` regardless of cwd

import numpy as np
import cv2
import imageio
from tqdm import tqdm

import torch
from scene.gaussian_renderer import GaussianModel, render
from scene import Scene

from utils.general_utils import format_output, set_seed
from utils.image_utils import to8b
from utils.graphics_utils import process_view

from argparse import ArgumentParser
from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args, merge_hparams

if __name__ == "__main__":

    parser = ArgumentParser(description="Render a demo video sweeping the camera from elev=-45 to elev=+45 across one frozen frame")
    modelParam = ModelParams()
    pipelineParam = PipelineParams()
    modelHiddenParam = ModelHiddenParams()

    modelParam.register(parser, set_default_none=True)
    pipelineParam.register(parser, set_default_none=True)
    modelHiddenParam.register(parser, set_default_none=True)

    parser.add_argument("--configs", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--expname", type=str, default="")

    parser.add_argument("--frame_idx", type=int, default=None, help="index into the video split to freeze and sweep; omit to sweep the whole sequence instead (frame 0 -> elev_min, last frame -> elev_max)")
    parser.add_argument("--elev_min", type=float, default=45, help="elevation at frame 0 (whole-sequence mode) or the first swept frame (frozen-frame mode)")
    parser.add_argument("--elev_max", type=float, default=-45, help="elevation at the last frame")
    parser.add_argument("--num_frames", type=int, default=90, help="number of rendered frames in the sweep (frozen-frame mode only; whole-sequence mode always uses every video frame)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--label", action="store_true", help="burn in the current elevation angle on each frame")
    parser.add_argument("--out_path", type=str, default=None, help="default: <model_path>/sideview_demo/ours_<iter>/sweep_frame<frame_idx>.mp4 or sweep_sequence.mp4 (individual PNGs go in a sibling renders_<name>/ folder)")

    args = parser.parse_args(sys.argv[1:])
    if args.expname:
        if not args.model_path:
            args.model_path = os.path.join("./output/", args.expname)
        if not args.configs:
            args.configs = os.path.join("./arguments/", args.expname + ".py")

    # get_combined_args rebuilds the namespace from the trained checkpoint's saved cfg_args,
    # only overlaying cmdline values where they're not None -- since cfg_args never had these
    # sweep-only args to begin with, any of them left at their (None-able) cmdline default
    # would vanish from the merged namespace entirely rather than fall back to that default.
    # Stash them beforehand and reapply after, so they always survive regardless of cfg_args.
    sweep_args = {
        "frame_idx": args.frame_idx, "elev_min": args.elev_min, "elev_max": args.elev_max,
        "num_frames": args.num_frames, "fps": args.fps, "label": args.label, "out_path": args.out_path,
    }

    args = get_combined_args(args)
    if args.configs and os.path.isfile(args.configs):
        import mmcv
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    for k, v in sweep_args.items():
        setattr(args, k, v)

    format_output(args.quiet)
    set_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    modelParam = modelParam.extract(args)
    modelHiddenParam = modelHiddenParam.extract(args)
    pipelineParam = pipelineParam.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(modelParam.sh_degree, modelHiddenParam)
        scene = Scene(
            modelParam,
            gaussians,
            load_iteration=args.iteration,
            load_coarse=modelParam.no_fine,
        )

        bg_color = [1, 1, 1] if modelParam.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        video_views = scene.getVideoViews()
        stage = "coarse" if modelParam.no_fine else "fine"

        def frame_distance(view):
            depth = view.original_depth
            depth = depth.cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
            valid_depth = depth[depth > 0]
            if valid_depth.size == 0:
                raise ValueError(f"frame {view.uid} has no valid (positive) depth values")
            return float(np.median(valid_depth))

        def render_at(view, elev, distance):
            view_new = process_view(view, 0, float(elev), distance)
            rendering = render(view_new, gaussians, pipelineParam, background, stage=stage)
            image = np.ascontiguousarray(to8b(rendering["render"].cpu()).transpose(1, 2, 0))  # CHW -> HWC, RGB
            if args.label:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(image_bgr, f"elev: {elev:+.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return image

        if args.frame_idx is not None:
            # frozen-frame mode: one reconstruction, --num_frames output frames sweeping elev.
            # step index (not the frozen frame's true index) names the output PNGs, since
            # every output here shares the same underlying frame -- only elev varies.
            frame_idx = args.frame_idx
            view = video_views[frame_idx]
            distance = frame_distance(view)
            elevs = np.linspace(args.elev_min, args.elev_max, args.num_frames)
            items = [(step, elev) for step, elev in enumerate(elevs)]
            frames = [render_at(view, elev, distance) for _, elev in tqdm(items, desc=f"Sweeping frame {frame_idx}")]
            out_name = f"sweep_frame{frame_idx}"
        else:
            # whole-sequence mode: every video frame gets its own evenly spaced elev,
            # frame 0 -> elev_min, last frame -> elev_max; output PNGs are named by the
            # frame's true index (view.uid), matching render.py/sideview.py's convention.
            # The pivot distance is fixed to frame 0's depth for every frame -- recomputing
            # it per-frame would move the orbit's pivot point as the video plays, making the
            # camera path jump around instead of orbiting smoothly.
            distance = frame_distance(video_views[0])
            elevs = np.linspace(args.elev_min, args.elev_max, len(video_views))
            items = [(view.uid, elev) for view, elev in zip(video_views, elevs)]
            frames = [render_at(view, elev, distance) for view, elev in tqdm(list(zip(video_views, elevs)), desc="Sweeping whole sequence")]
            out_name = "sweep_sequence"

        out_dir = os.path.dirname(args.out_path) if args.out_path else os.path.join(
            modelParam.model_path, "sideview_demo", "ours_{}".format(scene.loaded_iter)
        )
        out_path = args.out_path or os.path.join(out_dir, f"{out_name}.mp4")
        renders_dir = os.path.join(out_dir, f"renders_{out_name}")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(renders_dir, exist_ok=True)

        for (idx, elev), image in zip(items, frames):
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(renders_dir, f"{idx:05d}_elev{elev:+.1f}.png"), image_bgr)

        imageio.mimwrite(out_path, frames, fps=args.fps, quality=8)
        print(f"Saved {out_path}")
        print(f"Saved {len(frames)} frames to {renders_dir}")
