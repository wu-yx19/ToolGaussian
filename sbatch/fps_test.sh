#!/bin/bash
#SBATCH --job-name=fps_test
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/%j_fps_test.log
#SBATCH --error=logs/%j_error.log

mkdir -p logs
IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"
EXPNAME="${1:-endonerf/cutting-noaniso-nodepth}"

# --skip_test drops the test-set render; sideview rendering stays off by default (no
# --sideview_on_test/--elev/--frame_stride passed); video itself can't be skipped since
# --render_test's 50x timing loop is nested inside render.py's video render_set call
apptainer exec --nv $IMAGE_PATH /bin/bash -c "
    cd $PROJECT_DIR
    python render.py --expname $EXPNAME --skip_test --render_test
"
