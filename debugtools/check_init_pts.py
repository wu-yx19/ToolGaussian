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

print("\nFAILED: " + ", ".join(failures) if failures else "\nAll checks passed.")
raise SystemExit(1 if failures else 0)
