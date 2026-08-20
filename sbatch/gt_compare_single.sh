#!/bin/bash

#SBATCH --job-name=TG_gt_compare
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:03:00
#SBATCH --output=logs/%j_gt_compare.log
#SBATCH --error=logs/%j_error.log

# Pure numpy/cv2 (no CUDA needed), unlike warpback_single.sh -- runs on a CPU
# partition instead of requesting a GPU node it wouldn't use. Requires sideview.py
# (--save_depth --save_meta) and warpback.py to have already been run for this
# expname/elev, and render.py to have been run at least once (for its gt/masks
# output warpback_gt_compare.py reads).

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"
EXPNAME="${1:-endonerf/pulling}"
ELEV="${2:-20}"

echo "Working Directory: $PROJECT_DIR"
echo "Expname: $EXPNAME"
echo "Elev: $ELEV"
echo "Starting at: $(date)"

apptainer exec $IMAGE_PATH /bin/bash << EOF
    set -e
    cd $PROJECT_DIR

    echo "Comparing warpback vs ground truth: \$(date)"
    python warpback_gt_compare.py --expname $EXPNAME --elev $ELEV
EOF

echo "Finished at: $(date)"
