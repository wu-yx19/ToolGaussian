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
ELEV="${3:-5 10 15 20 30 45}"
BASELINE="${4:-endonerf/cutting-noaniso-nodepth}"  # compared against via debugtools/compare_experiments.py -- the controlled reference (pruning fixes kept, depth/aniso off), not the differently-trained cutting checkpoint; pass endonerf/pulling-noaniso-nodepth for pulling-family runs
CONFIG_PATH="./arguments/${EXPNAME}.py"     # train_eval.py's own default when --configs is unset

echo "Working Directory: $PROJECT_DIR"
echo "Dataset path: $DATA_PATH"
echo "Elev: $ELEV"
echo "Baseline: $BASELINE"
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
    # warp_to_source.py needs; render.py's own train/test/video pass also produces the gt/masks
    # output score_against_gt.py reads (from the video split, unaffected by --sideview_on_test).
    # --sideview_on_test sources sideview frames from the held-out test views instead of every
    # Nth video frame, matching debugtools/compare_experiments.py's "original view" comparison group.
    echo "Rendering started: \$(date)"
    python render.py --expname $EXPNAME --sideview_on_test --elev $ELEV --scale_check --save_depth --save_meta

    echo "Warping back to original view: \$(date)"
    python warp_to_source.py --expname $EXPNAME --elev $ELEV

    echo "Comparing warpback vs ground truth: \$(date)"
    python score_against_gt.py --expname $EXPNAME --elev $ELEV

    echo "Evaluation started: \$(date)"
    python evaluate.py --expname $EXPNAME

    # must run after evaluate.py -- compare_experiments.py's "orig" group reads evaluate.py's
    # own per_view.json, which doesn't exist until evaluate.py has run
    echo "Comparing against baseline $BASELINE: \$(date)"
    python debugtools/compare_experiments.py --exp1 $BASELINE --exp2 $EXPNAME --elev $ELEV
EOF

echo "Pipeline Finished at: $(date)"
