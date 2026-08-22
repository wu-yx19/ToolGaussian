#!/bin/bash

#SBATCH --job-name=TG_nova_compare
#SBATCH --partition=gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=logs/%j_nova_compare.log
#SBATCH --error=logs/%j_error.log

# Runs inside the NOVA container (nova_env.sif), not endo_env_new.sif -- this
# needs torch/timm/DINOv2, not the gaussian splatting stack. Requires
# sideview.py to have been run for this expname/elev, and render.py to have
# been run at least once (for its video/ours_<iteration>/gt output this reads).

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/nova_env.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"
NOVA_REPO="/home/groups/bdaniel/wyx/Projects/NOVA"
EXPNAME="${1:-endonerf/pulling}"
ELEV="${2:-20}"

echo "Working Directory: $PROJECT_DIR"
echo "Expname: $EXPNAME"
echo "Elev: $ELEV"
echo "Starting at: $(date)"

# Redirect model/weight caches off $HOME per Sherlock policy
export TORCH_HOME=$SCRATCH/cache/torch
export HF_HOME=$SCRATCH/cache/huggingface
mkdir -p "$TORCH_HOME" "$HF_HOME"

apptainer exec --nv \
    --bind /home/groups/bdaniel/wyx/Projects:/home/groups/bdaniel/wyx/Projects \
    --env NOVA_REPO="$NOVA_REPO" \
    "$IMAGE_PATH" /bin/bash << EOF
    set -e
    cd $PROJECT_DIR

    echo "Comparing sideview renders vs ground truth (NOVA): \$(date)"
    python nova_gt_compare.py --expname $EXPNAME --elev $ELEV
EOF

echo "Finished at: $(date)"
