#!/bin/bash
#SBATCH --job-name=check_init_pts
#SBATCH --partition=normal
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j_check_init_pts.log
#SBATCH --error=logs/%j_error.log

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

apptainer exec $IMAGE_PATH /bin/bash -c "cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR python3 debugtools/check_init_pts.py"
