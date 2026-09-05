#!/bin/bash

#SBATCH --job-name=TG_scared
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:16GB
#SBATCH --mem=16G                     # d2k1 (88 frames) measured at 10.8GB; scales with frame count
#SBATCH --time=0:10:00                # d2k1 measured at 4.5min; all 5 scenes are 80-99 frames
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
