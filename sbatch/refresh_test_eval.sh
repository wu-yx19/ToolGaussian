#!/bin/bash
#SBATCH --job-name=refresh_test_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=logs/%j_refresh_test_eval.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs
IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"
EXPNAME="${1:-endonerf/cutting}"

apptainer exec --nv $IMAGE_PATH /bin/bash -c "
    cd $PROJECT_DIR
    set -e
    echo 'Re-rendering test set with true frame indices:' \$(date)
    python render.py --expname $EXPNAME --skip_video --scale_check
    echo 'Re-running evaluate.py:' \$(date)
    python evaluate.py --expname $EXPNAME
"
