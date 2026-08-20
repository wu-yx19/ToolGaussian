import os
import sys
import tempfile

import mmcv
from argparse import ArgumentParser, Namespace
from arguments import (
    ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams,
    TestEvalParams, SideviewParams, RuntimeParams, merge_hparams,
)

with open("output/endonerf/cutting/cfg_args") as f:
    cutting_ns = eval(f.read().strip(), {"Namespace": Namespace})

parser = ArgumentParser()
for grp in [ModelParams(), OptimizationParams(), PipelineParams(), ModelHiddenParams(),
            TestEvalParams(), SideviewParams(), RuntimeParams()]:
    grp.register(parser)
defaults_args = parser.parse_args([])

config_path = sys.argv[1] if len(sys.argv) > 1 else "arguments/endonerf/cutting-sidereg.py"
cfg = mmcv.Config.fromfile(config_path)
merged = merge_hparams(defaults_args, cfg)

merged_d = vars(merged)
cutting_d = vars(cutting_ns)
skip = {"model_path", "expname", "configs", "port", "source_path"}
diffs = []
for k in cutting_d:
    if k in skip:
        continue
    mv = merged_d.get(k, "<MISSING>")
    cv = cutting_d[k]
    if mv != cv:
        diffs.append((k, cv, mv))

print(f"DIFFS between cutting's real cfg_args and {config_path} merged onto current defaults:")
for k, cv, mv in diffs:
    print(f"  {k}: cutting={cv!r}  new={mv!r}")
if not diffs:
    print("  (none)")
