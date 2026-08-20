#!/bin/bash
#SBATCH --job-name=compare_warpback
#SBATCH --partition=normal
#SBATCH --time=0:02:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%j_compare_warpback.log
#SBATCH --error=logs/%j_error.log

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"
EXP1="${1:-endonerf/cutting}"
EXP2="${2:-endonerf/cutting-depthreg}"

apptainer exec $IMAGE_PATH /bin/bash -c "cd $PROJECT_DIR && python3 debugtools/compare_warpback.py --exp1 $EXP1 --exp2 $EXP2"
