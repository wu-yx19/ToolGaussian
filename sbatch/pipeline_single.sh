#!/bin/bash

#SBATCH --job-name=TG_pipeline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:16GB
#SBATCH --time=0:20:00                # train + render + evaluate
#SBATCH --output=logs/%j_pipeline.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

GPU_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_gpu.log"
EXPNAME="endonerf/cutting"

echo "Working Directory: $PROJECT_DIR"
echo "Starting Pipeline at: $(date)"

apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    set -e
    cd $PROJECT_DIR

    echo "Start GPU logging"

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > $GPU_LOG &

    GPU_MONITOR_PID=\$!
    trap 'kill \$GPU_MONITOR_PID 2>/dev/null' EXIT

    echo "Training started: \$(date)"
    python train.py --expname $EXPNAME --no-log_file

    echo "Rendering started: \$(date)"
    python render.py --expname $EXPNAME --frame_stride 20 --elev 10

    echo "Evaluation started: \$(date)"
    python evaluate.py --expname $EXPNAME
EOF

echo "Pipeline Finished at: $(date)"
