#!/bin/bash

#SBATCH --job-name=TG_sideview
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=0:07:00
#SBATCH --output=logs/%j_sideview.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

GPU_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_gpu.log"
EXPNAME="${1:-endonerf/cutting}"

echo "Working Directory: $PROJECT_DIR"
echo "Expname: $EXPNAME"
echo "Starting Sideview Rendering at: $(date)"

apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    cd $PROJECT_DIR

    echo "Start GPU logging"

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > $GPU_LOG &

    GPU_MONITOR_PID=\$!

    # single sideview.py invocation loads the model once and loops over all three
    # elevations internally, instead of reloading it per elev
    python sideview.py --expname $EXPNAME --elev 10 20 30 --frame_stride 20 --save_depth

    echo "Stop GPU logging"
    kill \$GPU_MONITOR_PID
EOF

echo "Sideview Rendering Finished at: $(date)"
