#!/bin/bash

#SBATCH --job-name=TG_pipeline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=0:20:00                # train + render (incl. sideview) + warpback + compare + evaluate
#SBATCH --output=logs/%j_pipeline.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

GPU_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_gpu_usage.log"
MEM_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_mem_usage.log"
SAMPLE_INTERVAL=5                     # seconds between usage samples
EXPNAME="${1:-endonerf/cutting}"
DATA_PATH="${2:-data/endonerf/cutting}"     # config name may differ from the underlying dataset dir (e.g. cutting-nosv -> data/endonerf/cutting)
ELEV="${3:-10 20 30}"
CONFIG_PATH="./arguments/${EXPNAME}.py"     # train_eval.py's own default when --configs is unset

echo "Working Directory: $PROJECT_DIR"
echo "Dataset path: $DATA_PATH"
echo "Elev: $ELEV"
echo "Config path: $CONFIG_PATH"
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
    # unique port per job -- concurrent pipeline_single.sh jobs landing on the same node
    # otherwise collide on the network_gui's fixed default port (6009) and crash on bind()
    python train_eval.py --expname $EXPNAME --source_path $DATA_PATH --no-log_file --port $((6009 + SLURM_JOB_ID % 1000))

    # --save_depth/--save_meta also render sideviews (render.py already calls sideview.py's
    # render_frame_views internally) and write the raw depth (.npy) + camera params (.json)
    # warpback.py needs; render.py's own train/test/video pass also produces the gt/masks
    # output warpback_gt_compare.py reads.
    echo "Rendering started: \$(date)"
    python render.py --expname $EXPNAME --frame_stride 20 --elev $ELEV --scale_check --save_depth --save_meta

    echo "Warping back to original view: \$(date)"
    python warpback.py --expname $EXPNAME --elev $ELEV

    echo "Comparing warpback vs ground truth: \$(date)"
    python warpback_gt_compare.py --expname $EXPNAME --elev $ELEV

    echo "Evaluation started: \$(date)"
    python evaluate.py --expname $EXPNAME
EOF

echo "Pipeline Finished at: $(date)"
