#!/bin/bash
# run_ddp.sh
#
# Example: sbatch run_ddp.sh 32 2

#SBATCH --job-name=lab5_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 4 processes per node
#SBATCH --gres=gpu:4                 # 4 GPUs
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

# --- Modules (EDIT to match Greene) ---
module purge
module load anaconda3
module load cuda/11.7      # or the version available

# --- Conda env (EDIT) ---
source activate hpml        # or `conda activate hpml`

mkdir -p logs

BATCH_SIZE_PER_GPU=${1:-32}
EPOCHS=${2:-2}

echo "Running Lab 5 DDP with batch_size_per_gpu=${BATCH_SIZE_PER_GPU}, epochs=${EPOCHS}"

torchrun --standalone --nproc_per_node=4 ddp_train.py \
    --data_path ./data \
    --batch_size "${BATCH_SIZE_PER_GPU}" \
    --epochs "${EPOCHS}" \
    --measure_epoch 1