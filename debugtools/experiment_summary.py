#
# experiment_summary.py
#
# Consolidate, per experiment, the 9 metrics scattered across separate files:
#   - results.json: SSIM, PSNR, PSNR*, LPIPS, FLIP, RMSE
#   - logs/*_pipeline.log: training_time (Training started -> Rendering started)
#   - logs/*_gpu_usage.log: peak GPU memory (MiB) during that same training window
#   - logs/*_fps_test.log: FPS (render.py --render_test)
#
# Scans output/endonerf/*/ for experiments with a results.json, and writes one combined
# file: output/endonerf/experiment_summary.json.
#
# Usage:
#   python debugtools/experiment_summary.py
#

import os
import re
import csv
import glob
import json
from datetime import datetime
from argparse import ArgumentParser

TIMESTAMP_RE = re.compile(r"^(.+?):\s*(\w{3} \w{3} +\d+ \d{2}:\d{2}:\d{2}) \w+ (\d{4})$")
FPS_RE = re.compile(r"FPS:\s*([\d.]+)")


def parse_event(line, label):
    m = TIMESTAMP_RE.match(line.rstrip("\n"))
    if not m or not m.group(1).startswith(label):
        return None
    ts_no_tz, year = m.group(2), m.group(3)
    # drop the timezone abbreviation (PDT/PST/...) -- unreliable to parse portably with
    # %Z, and unnecessary since we only diff two timestamps from the same log
    return datetime.strptime(f"{ts_no_tz} {year}", "%a %b %d %H:%M:%S %Y")


def training_window(pipeline_log_path):
    start = end = None
    with open(pipeline_log_path) as f:
        for line in f:
            start = start or parse_event(line, "Training started")
            end = end or parse_event(line, "Rendering started")
    return start, end


def peak_gpu_mem(gpu_log_path, start, end):
    if not os.path.isfile(gpu_log_path):
        return None
    max_mem = None
    with open(gpu_log_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            mem = int(row[3].strip().rstrip(" MiB"))
            max_mem = mem if max_mem is None else max(max_mem, mem)
    return max_mem


def parse_fps(path):
    with open(path) as f:
        for line in f:
            m = FPS_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def collect_one(exp_dir):
    results_path = os.path.join(exp_dir, "results.json")
    if not os.path.isfile(results_path):
        return None
    with open(results_path) as f:
        metrics = next(iter(json.load(f).values()))  # {"ours_<iter>": {...}} -> {...}

    entry = {
        "SSIM": metrics.get("SSIM"),
        "PSNR": metrics.get("PSNR"),
        "PSNR*": metrics.get("PSNR*"),
        "LPIPS": metrics.get("LPIPS"),
        "FLIP": metrics.get("FLIP"),
        "RMSE": metrics.get("RMSE"),
        "training_time_sec": None,
        "peak_gpu_mem_mib": None,
        "fps": None,
    }

    pipeline_logs = glob.glob(os.path.join(exp_dir, "logs", "*_pipeline.log"))
    if pipeline_logs:
        # if an experiment was retrained more than once, use the most recently modified log
        pipeline_log = max(pipeline_logs, key=os.path.getmtime)
        start, end = training_window(pipeline_log)
        if start and end:
            entry["training_time_sec"] = (end - start).total_seconds()
            job_id = os.path.basename(pipeline_log).split("_")[0]
            gpu_log = os.path.join(exp_dir, "logs", f"{job_id}_gpu_usage.log")
            entry["peak_gpu_mem_mib"] = peak_gpu_mem(gpu_log, start, end)

    fps_logs = glob.glob(os.path.join(exp_dir, "logs", "*_fps_test.log"))
    if fps_logs:
        fps_log = max(fps_logs, key=os.path.getmtime)
        entry["fps"] = parse_fps(fps_log)

    return entry


def main():
    parser = ArgumentParser(description="Consolidate results.json + training time/GPU/FPS into one summary file")
    parser.add_argument("--root", default=os.path.join("output", "endonerf"))
    parser.add_argument("--out_path", default=None, help="default: <root>/experiment_summary.json")
    args = parser.parse_args()
    out_path = args.out_path or os.path.join(args.root, "experiment_summary.json")

    summary = {}
    for exp_dir in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        expname = os.path.basename(exp_dir)
        entry = collect_one(exp_dir)
        if entry is not None:
            summary[expname] = entry

    if not summary:
        print("No experiments with results.json found.")
        return

    cols = ["SSIM", "PSNR", "PSNR*", "LPIPS", "FLIP", "RMSE", "training_time_sec", "peak_gpu_mem_mib", "fps"]
    print(f"{'experiment':40s} " + " ".join(f"{c:>14s}" for c in cols))
    for expname, entry in summary.items():
        row = []
        for c in cols:
            v = entry[c]
            row.append("--" if v is None else f"{v:.4f}")
        print(f"{expname:40s} " + " ".join(f"{v:>14s}" for v in row))

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
