#!/bin/bash

#SBATCH --job-name=TG_scared
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:16GB
#SBATCH --mem=32G                     # ~90 frames of 1280x1024 held in RAM during point init
#SBATCH --time=1:00:00                # 2000 iters, but 4x the pixels of endonerf (which fits 6000 in 20min)
#SBATCH --output=logs/%j_train_scared.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

EXPNAME="${1:-scared/d1k1}" # scared/d1k1 | d2k1 | d3k1 | d6k1 | d7k1
PORT="${2:-6009}"

GPU_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_gpu.log"

echo "Working Directory: $PROJECT_DIR"
echo "Experiment: $EXPNAME"
echo "Starting Training at: $(date)"

apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    cd $PROJECT_DIR

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > $GPU_LOG &

    GPU_MONITOR_PID=\$!

    # source_path, configs and model_path are all derived from --expname
    python train_eval.py --expname $EXPNAME --port $PORT

    kill \$GPU_MONITOR_PID
EOF

echo "Training Finished at: $(date)"
