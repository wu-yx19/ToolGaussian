#!/bin/bash

#SBATCH --job-name=TG_pipeline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:16GB
#SBATCH --mem=16G
#SBATCH --time=0:20:00                # train + render + evaluate
#SBATCH --output=logs/%j_pipeline.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

USAGE_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_usage.log"
SAMPLE_INTERVAL=5                     # seconds between usage samples
EXPNAME="endonerf/cutting"

echo "Working Directory: $PROJECT_DIR"
echo "Starting Pipeline at: $(date)"

apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    set -e
    cd $PROJECT_DIR

    echo "Start GPU + memory logging"

    echo "timestamp, gpu_util_pct, gpu_mem_util_pct, gpu_mem_used_MiB, host_mem_used_MiB, host_mem_total_MiB" > $USAGE_LOG
    ( while true; do
        GPU_STATS=\$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits)
        HOST_MEM=\$(free -m | awk '/^Mem:/ {print \$3", "\$2}')
        echo "\$(date '+%Y/%m/%d %H:%M:%S'), \$GPU_STATS, \$HOST_MEM"
        sleep $SAMPLE_INTERVAL
    done ) >> $USAGE_LOG &

    USAGE_MONITOR_PID=\$!
    trap 'kill \$USAGE_MONITOR_PID 2>/dev/null' EXIT

    echo "Training started: \$(date)"
    python train.py --expname $EXPNAME --no-log_file

    echo "Rendering started: \$(date)"
    python render.py --expname $EXPNAME --frame_stride 20 --elev 10

    echo "Evaluation started: \$(date)"
    python evaluate.py --expname $EXPNAME
EOF

echo "Pipeline Finished at: $(date)"
