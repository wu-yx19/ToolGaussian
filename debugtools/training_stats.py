#
# training_stats.py
#
# Parse a pipeline_single.sh log (and its sibling *_gpu_usage.log) to report how long
# training took (Training started -> Rendering started) and the peak GPU utilization /
# memory usage during that same window.
#
# By default scans output/endonerf/*/logs/*_pipeline.log (i.e. every experiment whose logs
# have been moved into its own output folder); pass explicit paths/expnames to restrict it.
#
# Usage:
#   python debugtools/training_stats.py
#   python debugtools/training_stats.py --expname endonerf/cutting-depthreg-aniso1e5-depth002
#   python debugtools/training_stats.py --log_path output/endonerf/cutting/logs/12345_pipeline.log
#

import os
import re
import csv
import glob
from datetime import datetime
from argparse import ArgumentParser

TIMESTAMP_RE = re.compile(r"^(.+?):\s*(\w{3} \w{3} +\d+ \d{2}:\d{2}:\d{2}) \w+ (\d{4})$")


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


def peak_gpu_usage(gpu_log_path, start, end):
    # nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used
    # --format=csv: "2026/08/21 21:25:47.567, 0 %, 0 %, 1 MiB"
    if not os.path.isfile(gpu_log_path):
        return None, None
    max_util, max_mem = None, None
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
            util = int(row[1].strip().rstrip(" %"))
            mem = int(row[3].strip().rstrip(" MiB"))
            max_util = util if max_util is None else max(max_util, util)
            max_mem = mem if max_mem is None else max(max_mem, mem)
    return max_util, max_mem


def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def resolve_log_paths(args):
    if args.log_path:
        return args.log_path
    if args.expname:
        return [
            p for exp in args.expname
            for p in glob.glob(os.path.join("output", exp, "logs", "*_pipeline.log"))
        ]
    return sorted(glob.glob(os.path.join("output", "endonerf", "*", "logs", "*_pipeline.log")))


def main():
    parser = ArgumentParser(description="Report training time and peak GPU usage from pipeline_single.sh logs")
    parser.add_argument("--expname", nargs="+", help="e.g. endonerf/cutting-depthreg-aniso1e5-depth002 (looks under output/<expname>/logs/)")
    parser.add_argument("--log_path", nargs="+", help="explicit path(s) to *_pipeline.log, overrides --expname")
    args = parser.parse_args()

    log_paths = resolve_log_paths(args)
    if not log_paths:
        print("No pipeline logs found.")
        return

    print(f"{'experiment':45s} {'training time':>15s} {'peak GPU util':>15s} {'peak GPU mem':>15s}")
    for path in log_paths:
        # output/endonerf/<expname>/logs/<job>_pipeline.log -> <expname>
        expname = os.path.basename(os.path.dirname(os.path.dirname(path)))
        job_id = os.path.basename(path).split("_")[0]
        start, end = training_window(path)
        duration = format_duration((end - start).total_seconds()) if start and end else "--"

        gpu_log_path = os.path.join(os.path.dirname(path), f"{job_id}_gpu_usage.log")
        max_util, max_mem = peak_gpu_usage(gpu_log_path, start, end)
        util_str = f"{max_util}%" if max_util is not None else "--"
        mem_str = f"{max_mem} MiB" if max_mem is not None else "--"

        print(f"{expname:45s} {duration:>15s} {util_str:>15s} {mem_str:>15s}")


if __name__ == "__main__":
    main()
