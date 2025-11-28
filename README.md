# gradient-gossip-network

Distributed CIFAR-10 training with ResNet and DDP, plus post-hoc quantization experiments for HPML Lab 5.

---

## Project Overview

This repo contains:

- **Part A – DDP Experiments**
  - Time vs batch size on **1 GPU** (Q1)
  - Speedup on **1 / 2 / 4 GPUs** (Q2)
  - Compute vs communication breakdown (Q3.1)
  - Effective bandwidth from ring all-reduce model (Q3.2)
  - Large-batch training behavior (Q4.1–Q4.2)
- **Part B – Quantization Experiments**
  - Weight / activation / bias histograms & 8-bit quantization
  - Accuracy impact of quantization

---

## Repo Structure

```text
.
├── c2.py                  # Lab 2 ResNet-18 definition (BasicBlock, ResNet)
├── ddp_train.py           # Main training script (1-GPU + DDP)
├── run_ddp.sh             # Slurm script for 4-GPU runs on Greene
├── quantization.ipynb     # Colab-style notebook for Part B (weights/acts/bias)
├── logs/                  # Saved logs from experiments
└── results/               # Tables/plots used in the report
```

> `ddp_train.py` imports `ResNet` and `BasicBlock` directly from `c2.py`.

---

## Environment

- Python 3.9+
- PyTorch with CUDA
- TorchVision
- Slurm cluster with NCCL for multi-GPU (e.g., NYU Greene)

Example (conda):

```bash
conda create -n hpml-lab5 python=3.9 -y
conda activate hpml-lab5
pip install torch torchvision
```

---

## Part A – Reproducing DDP Experiments

### 1. Q1 – Time vs Batch Size (1 GPU)

Measure **compute time per epoch** as batch size grows:

```bash
# Example: batch size 32, 1 GPU, 2 epochs, measure epoch 1
torchrun --standalone --nproc_per_node=1 ddp_train.py \
  --data_path ./data \
  --batch_size 32 \
  --epochs 2 \
  --measure_epoch 1 \
  --no_ddp > logs/bs32_1gpu.log
```

Repeat for batch sizes: `32, 128, 512, ...` (until OOM).  
Use the printed line:

```text
[TIME] epoch=1 | total_time_sec=... | compute_time_sec=...
```

Use `compute_time_sec` for Q1 plots/tables.

---

### 2. Q2 – Speedup (1 / 2 / 4 GPUs)

Keep **batch size per GPU fixed** (weak scaling). Example for `B = 32`:

```bash
# 1 GPU baseline
torchrun --standalone --nproc_per_node=1 ddp_train.py \
  --data_path ./data \
  --batch_size 32 \
  --epochs 2 \
  --measure_epoch 1 \
  --no_ddp > logs/bs32_1gpu.log

# 2 GPUs
torchrun --standalone --nproc_per_node=2 ddp_train.py \
  --data_path ./data \
  --batch_size 32 \
  --epochs 2 \
  --measure_epoch 1 > logs/bs32_2gpu.log

# 4 GPUs
torchrun --standalone --nproc_per_node=4 ddp_train.py \
  --data_path ./data \
  --batch_size 32 \
  --epochs 2 \
  --measure_epoch 1 > logs/bs32_4gpu.log
```

From `[TIME]` lines, use `total_time_sec` to compute:

\[
\text{Speedup}(N) = T_{1\text{GPU}} / T_{N\text{GPU}}
\]

Fill **Table 1** for each batch size.

---

### 3. Q3.1 – Compute vs Communication Time

For each (batch size per GPU, GPU count):

- Take `compute_time_sec` and `total_time_sec` from the same run.
- Approximate:

```text
compute_time  ≈ compute_time_sec
comm_time     ≈ total_time_sec - compute_time_sec
```

Fill **Table 2** with compute / comm times per epoch.

---

### 4. Q3.2 – Ring All-Reduce Bandwidth

`ddp_train.py` prints an approximate gradient size:

```text
[MODEL] total gradient size ≈ X.XX MB
```

For each 2-GPU and 4-GPU config:

- Use gradient size `N` (bytes)
- Use measured `comm_time`
- Effective bandwidth:

\[
\text{BW}_{\text{eff}} = \frac{2 (P-1)}{P} \cdot \frac{N}{T_{\text{comm}}}
\]

Convert to GB/s and compare to theoretical link BW to get utilization for **Table 3**.

---

### 5. Q4.1 – Large-Batch Training

1. From Q1, find the **largest per-GPU batch size** that fits on 1 GPU (`B_max`).
2. Run 4-GPU training:

```bash
torchrun --standalone --nproc_per_node=4 ddp_train.py \
  --data_path ./data \
  --batch_size B_MAX \
  --epochs 5 \
  --measure_epoch 1 > logs/bs${B_MAX}_4gpu_e5.log
```

3. Use the `[EPOCH] 4 | loss=... | acc=...` line as **epoch 5** stats.
4. Compare against your Lab 2 run (batch size 128) in the report.

---

## Part B – Quantization Notebook

In `quantization.ipynb`:

1. **Train baseline CNN** on CIFAR-10; log FP32 test accuracy.
2. **Q1 – Weight histograms:** visualize conv/fc weights.
3. **Q2 – Weight quantization:** uniform 8-bit quantization, then re-evaluate accuracy.
4. **Q3 – Activation histograms:** capture and plot activations via forward hooks.
5. **Q4 – Activation quantization:** quantize activations + weights, measure accuracy drop.
6. **Q5 – Bias quantization:** optionally quantize biases and report final accuracy.

Export plots and key numbers into your PDF report.

---

## Notes

- `ddp_train.py` uses epoch 0 as warmup and **measures epoch 1** by default.
- All reported times are **per epoch**.
- Logs in `logs/` are the only source of truth for filling tables and making plots.
