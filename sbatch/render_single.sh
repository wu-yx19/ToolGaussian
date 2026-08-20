#!/bin/bash

#SBATCH --job-name=TG_render
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:16GB
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/%j_render.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

GPU_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_gpu.log"
FILE_NAME="${1:-endonerf/cutting}"

echo "Working Directory: $PROJECT_DIR"
echo "Starting Training at: $(date)"

apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    cd $PROJECT_DIR

    echo "Start GPU logging"

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > $GPU_LOG &

    GPU_MONITOR_PID=\$!

    python render.py --expname $FILE_NAME

    echo "Stop GPU logging"
    kill \$GPU_MONITOR_PID
EOF

echo "Rendering Finished at: $(date)"
