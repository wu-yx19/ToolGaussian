#!/bin/bash

#SBATCH --job-name=TG_build_alpha
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=logs/%j_build_alpha.log
#SBATCH --error=logs/%j_build_alpha_error.log

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"
EXPNAME="${1:-endonerf/pulling}"

echo "Working Directory: $PROJECT_DIR"
echo "Expname: $EXPNAME"
echo "Starting build+smoke-test at: $(date)"

# Builds the modified rasterizer and installs it to the user site-packages
# (~/.local), which Python resolves ahead of the container's baked-in
# /opt/conda site-packages -- the shared sandbox image is never touched.
apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    set -e
    cd $PROJECT_DIR/submodules/depth-diff-gaussian-rasterization

    echo "Building and installing to --user site-packages: \$(date)"
    TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6 9.0+PTX" pip install --user --no-build-isolation --force-reinstall --no-deps .

    cd $PROJECT_DIR

    echo "Verifying install location: \$(date)"
    python3 -c "import diff_gaussian_rasterization, os; print('Loaded from:', os.path.dirname(diff_gaussian_rasterization.__file__))"

    echo "Running smoke test: \$(date)"
    PYTHONPATH=$PROJECT_DIR python3 debugtools/check_accum_alpha.py --expname $EXPNAME
EOF

echo "Finished at: $(date)"
