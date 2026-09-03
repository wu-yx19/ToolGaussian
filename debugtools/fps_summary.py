#
# fps_summary.py
#
# Collect FPS numbers from *_fps_test.log files (written by sbatch/fps_test.sh, which runs
# render.py --render_test) and save a consolidated summary.
#
# By default scans output/endonerf/*/logs/*_fps_test.log; writes the result to
# output/endonerf/fps_summary.json (one entry per experiment) and prints a table.
#
# Usage:
#   python debugtools/fps_summary.py
#

import os
import re
import glob
import json
from argparse import ArgumentParser

FPS_RE = re.compile(r"FPS:\s*([\d.]+)")


def parse_fps(path):
    with open(path) as f:
        for line in f:
            m = FPS_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def main():
    parser = ArgumentParser(description="Collect and save FPS numbers from fps_test.sh logs")
    parser.add_argument("--log_glob", default=os.path.join("output", "endonerf", "*", "logs", "*_fps_test.log"))
    parser.add_argument("--out_path", default=os.path.join("output", "endonerf", "fps_summary.json"))
    args = parser.parse_args()

    results = {}
    for path in sorted(glob.glob(args.log_glob)):
        # output/endonerf/<expname>/logs/<job>_fps_test.log -> <expname>
        expname = os.path.basename(os.path.dirname(os.path.dirname(path)))
        fps = parse_fps(path)
        if fps is not None:
            results[expname] = fps

    if not results:
        print("No FPS results found.")
        return

    print(f"{'experiment':45s} {'FPS':>10s}")
    for expname, fps in results.items():
        print(f"{expname:45s} {fps:10.2f}")

    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_path}")


if __name__ == "__main__":
    main()
