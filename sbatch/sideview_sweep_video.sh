#!/bin/bash
#SBATCH --job-name=sweep_video
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/%j_sweep_video.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs
IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"
EXPNAME="${1:-endonerf/cutting-depthreg-aniso1e5-depth002}"
FRAME_IDX="${2:-}"
FRAME_FLAG=""
if [ -n "$FRAME_IDX" ]; then
    FRAME_FLAG="--frame_idx $FRAME_IDX"
fi

apptainer exec --nv $IMAGE_PATH /bin/bash -c "
    cd $PROJECT_DIR
    python debugtools/sideview_sweep_video.py --expname $EXPNAME $FRAME_FLAG --label
"
