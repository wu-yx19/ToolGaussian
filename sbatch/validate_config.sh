#!/bin/bash
#SBATCH --job-name=validate_cfg
#SBATCH --partition=normal
#SBATCH --time=0:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%j_validate.log
#SBATCH --error=logs/%j_error.log

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

apptainer exec $IMAGE_PATH /bin/bash -c "cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR python3 debugtools/validate_sidereg_config.py ${1:-arguments/endonerf/cutting-sidereg.py}"
