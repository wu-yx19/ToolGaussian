import os
import sys
import tempfile

import mmcv
from argparse import ArgumentParser, Namespace
from arguments import (
    ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams,
    TestEvalParams, SideviewParams, RuntimeParams, merge_hparams,
)

config_path = sys.argv[1] if len(sys.argv) > 1 else "arguments/endonerf/cutting-sidereg.py"
baseline_exp = sys.argv[2] if len(sys.argv) > 2 else "cutting"

with open(f"output/endonerf/{baseline_exp}/cfg_args") as f:
    baseline_ns = eval(f.read().strip(), {"Namespace": Namespace})

parser = ArgumentParser()
for grp in [ModelParams(), OptimizationParams(), PipelineParams(), ModelHiddenParams(),
            TestEvalParams(), SideviewParams(), RuntimeParams()]:
    grp.register(parser)
defaults_args = parser.parse_args([])

cfg = mmcv.Config.fromfile(config_path)
merged = merge_hparams(defaults_args, cfg)

merged_d = vars(merged)
baseline_d = vars(baseline_ns)
skip = {"model_path", "expname", "configs", "port", "source_path"}
diffs = []
for k in baseline_d:
    if k in skip:
        continue
    mv = merged_d.get(k, "<MISSING>")
    bv = baseline_d[k]
    if mv != bv:
        diffs.append((k, bv, mv))

print(f"DIFFS between {baseline_exp}'s real cfg_args and {config_path} merged onto current defaults:")
for k, bv, mv in diffs:
    print(f"  {k}: {baseline_exp}={bv!r}  new={mv!r}")
if not diffs:
    print("  (none)")
