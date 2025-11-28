# Q1: 1-GPU, no DDP, varying batch sizes
torchrun --standalone --nproc_per_node=1 ddp_train.py \
    --data_path ./data \
    --batch_size 32 \
    --epochs 2 \
    --measure_epoch 1 \
    --no_ddp > logs/bs32_1gpu.log

# Q2/Q3: 2 GPUs
torchrun --standalone --nproc_per_node=2 ddp_train.py \
    --data_path ./data \
    --batch_size 32 \
    --epochs 2 \
    --measure_epoch 1 > logs/bs32_2gpu.log

# Q2/Q3/Q4.1: 4 GPUs
torchrun --standalone --nproc_per_node=4 ddp_train.py \
    --data_path ./data \
    --batch_size 32 \
    --epochs 2 \
    --measure_epoch 1 > logs/bs32_4gpu.log

# Q4.1 with the largest per-GPU batch that fits:
torchrun --standalone --nproc_per_node=4 ddp_train.py \
    --data_path ./data \
    --batch_size <B_MAX> \
    --epochs 5 \
    --measure_epoch 1 > logs/bs${B_MAX}_4gpu_e5.log