#!/bin/bash

#SBATCH --job-name=TG_train           # 任务名
#SBATCH --partition=gpu               # 使用 gpu 分区
#SBATCH --gpus=1                      # 申请 1 块 GPU
#SBATCH --constraint=GPU_MEM:16GB     # 强制要求 16GB 显存的卡
#SBATCH --mem=16G                     # 申请 16G 内存
#SBATCH --time=0:20:00               # 预计运行时间 (根据需求调整)
#SBATCH --output=logs/%j_train.log    # 标准输出日志
#SBATCH --error=logs/%j_error.log     # 错误日志

mkdir -p logs

IMAGE_PATH="/home/groups/bdaniel/wyx/docker/endo_env_new.sif"
PROJECT_DIR="/home/groups/bdaniel/wyx/Projects/ToolGaussian"

GPU_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}_gpu.log"
FILE_NAME="hamlyn/hamlyn_seq1" # endonerf/cutting

echo "Working Directory: $PROJECT_DIR"
echo "Starting Training at: $(date)"

apptainer exec --nv $IMAGE_PATH /bin/bash << EOF
    cd $PROJECT_DIR

    echo "Start GPU logging"

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > $GPU_LOG &

    GPU_MONITOR_PID=\$!

    python train_eval.py --source_path data/$FILE_NAME \
        --port 6009 \
        --expname $FILE_NAME \
        --configs arguments/$FILE_NAME.py

    echo "Stop GPU logging"
    kill \$GPU_MONITOR_PID
EOF

echo "Training Finished at: $(date)"
