"""Check that init_pts is configurable for all three dataset loaders.

Covers the arg plumbing (register -> config merge -> extract) and then loads real endonerf and
hamlyn data to confirm get_init_pts() honors the requested count. Runs on CPU; no GPU needed,
since get_init_pts() does not build View objects.

  sbatch sbatch/check_init_pts.sh
"""

from argparse import ArgumentParser

import mmcv

from arguments import ModelParams, merge_hparams
from scene.datasets import EndoNeRF_Dataset, Hamlyn_Dataset

failures = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got} (want {want})")
    if not ok:
        failures.append(label)


print("[1] arg plumbing")
parser = ArgumentParser()
model_params = ModelParams()
model_params.register(parser)

args = parser.parse_args(["-s", "data/endonerf/cutting"])
check("default init_pts", args.init_pts, 30_000)

scared_args = merge_hparams(
    parser.parse_args(["-s", "data/scared/dataset_1/keyframe_1"]),
    mmcv.Config.fromfile("arguments/scared/d1k1.py"),
)
check("scared config merge", scared_args.init_pts, 20_000)
check("scared extract", model_params.extract(scared_args).init_pts, 20_000)

endonerf_args = merge_hparams(
    parser.parse_args(["-s", "data/endonerf/cutting"]),
    mmcv.Config.fromfile("arguments/endonerf/cutting-depthreg-aniso1e5-depth002.py"),
)
check("endonerf keeps default", endonerf_args.init_pts, 30_000)
check("CLI override", parser.parse_args(["-s", "x", "--init_pts", "12345"]).init_pts, 12345)

# an old checkpoint's cfg_args has no init_pts at all; scene/__init__.py must not raise
print("  ok   getattr fallback:", getattr(model_params.extract(parser.parse_args(["-s", "x"])), "nonexistent", 30_000))

print("\n[2] endonerf loader honors init_pts (loads real data)")
ds = EndoNeRF_Dataset("data/endonerf/cutting", init_pts=5_000)
check("endonerf get_init_pts", ds.get_init_pts()[0].shape[0], 5_000)

print("\n[3] hamlyn loader honors init_pts (loads real data)")
ds = Hamlyn_Dataset("data/hamlyn/hamlyn_seq1", init_pts=5_000)
check("hamlyn get_init_pts", ds.get_init_pts()[0].shape[0], 5_000)

# 640x480 = 307_200 pixels, so this exceeds the valid-pixel count and must fall back to
# replace=True instead of raising ValueError
ds = Hamlyn_Dataset("data/hamlyn/hamlyn_seq1", init_pts=400_000)
check("hamlyn oversubscribed", ds.get_init_pts()[0].shape[0], 400_000)

print("\n[4] omitting init_pts falls back to the loader default (what old checkpoints hit)")
ds = EndoNeRF_Dataset("data/endonerf/cutting")
check("endonerf loader default", ds.get_init_pts()[0].shape[0], 30_000)

print("\n[5] expname -> path resolution")
from argparse import Namespace
import os.path

from utils.general_utils import expname_to_source_path, resolve_expname_paths

check("scared/d1k1", expname_to_source_path("scared/d1k1"), "./data/scared/dataset_1/keyframe_1")
check("scared/d7k1", expname_to_source_path("scared/d7k1"), "./data/scared/dataset_7/keyframe_1")
check("scared/d1k1_mono", expname_to_source_path("scared/d1k1_mono"), "./data/scared/dataset_1/keyframe_1")
check("endonerf/cutting", expname_to_source_path("endonerf/cutting"), "./data/endonerf/cutting")
check("inferred path exists", os.path.isdir(expname_to_source_path("scared/d1k1")), True)

ns = Namespace(expname="scared/d1k1", model_path="", configs="", source_path="")
resolve_expname_paths(ns, infer_source_path=True)
check("model_path", ns.model_path, "./output/scared/d1k1")
check("configs", ns.configs, "./arguments/scared/d1k1.py")
check("source_path", ns.source_path, "./data/scared/dataset_1/keyframe_1")

ns = Namespace(expname="scared/d1k1", model_path="", configs="", source_path="data/custom")
resolve_expname_paths(ns, infer_source_path=True)
check("explicit source_path wins", ns.source_path, "data/custom")

ns = Namespace(expname="scared/d1k1", model_path="", configs="", source_path="")
resolve_expname_paths(ns)
check("source_path untouched by default", ns.source_path, "")

print("\nFAILED: " + ", ".join(failures) if failures else "\nAll checks passed.")
raise SystemExit(1 if failures else 0)
