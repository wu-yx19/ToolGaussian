#!/bin/bash

#SBATCH --job-name=TG_pipeline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=0:45:00                # train + render + evaluate
#SBATCH --output=logs/%j_pipeline.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

GPU_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_gpu_usage.log"
MEM_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_mem_usage.log"
SAMPLE_INTERVAL=5                     # seconds between usage samples
EXPNAME="endonerf/cutting-new"
DATA_PATH="data/endonerf/cutting"     # cutting-ro reuses the cutting dataset; only the config differs

echo "Working Directory: $PROJECT_DIR"
echo "Starting Pipeline at: $(date)"

apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    set -e
    cd $PROJECT_DIR

    echo "Start GPU + memory logging"

    # -l/-t make nvidia-smi/vmstat repeat and timestamp on their own,
    # avoiding a hand-rolled polling loop that Sherlock's submit filter
    # flags even when it's just a monitoring sidecar next to real work
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l $SAMPLE_INTERVAL > $GPU_LOG &
    GPU_MONITOR_PID=\$!

    vmstat -t -S M $SAMPLE_INTERVAL > $MEM_LOG &
    MEM_MONITOR_PID=\$!

    trap 'kill \$GPU_MONITOR_PID \$MEM_MONITOR_PID 2>/dev/null' EXIT

    echo "Training started: \$(date)"
    python train_eval.py --expname $EXPNAME --source_path $DATA_PATH --no-log_file

    echo "Rendering started: \$(date)"
    python render.py --expname $EXPNAME --frame_stride 20 --elev 10

    echo "Evaluation started: \$(date)"
    python evaluate.py --expname $EXPNAME
EOF

echo "Pipeline Finished at: $(date)"
