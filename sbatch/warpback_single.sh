#!/bin/bash

#SBATCH --job-name=TG_warpback
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=logs/%j_warpback.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

GPU_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_gpu.log"
EXPNAME="${1:-endonerf/pulling}"
ELEV="${2:-20}"

echo "Working Directory: $PROJECT_DIR"
echo "Expname: $EXPNAME"
echo "Starting Sideview + Warpback at: $(date)"

apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    set -e
    cd $PROJECT_DIR

    echo "Start GPU logging"

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > $GPU_LOG &

    GPU_MONITOR_PID=\$!

    # --save_depth saves the rendered depth; --save_meta additionally writes the raw depth
    # (.npy) + camera params (.json) the standalone warpback.py / warpback_gt_compare.py
    # tools need for their own post-hoc warp-back pass (no need for --warp_mode here)
    echo "Rendering sideviews: \$(date)"
    python sideview.py --expname $EXPNAME --save_depth --save_meta --frame_stride 20 --elev $ELEV

    echo "Warping back to original view: \$(date)"
    python warpback.py --expname $EXPNAME --elev $ELEV

    echo "Comparing warpback vs ground truth: \$(date)"
    python warpback_gt_compare.py --expname $EXPNAME --elev $ELEV

    echo "Stop GPU logging"
    kill \$GPU_MONITOR_PID
EOF

echo "Sideview + Warpback Finished at: $(date)"
