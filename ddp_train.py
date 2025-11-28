# ddp_train.py
#
# HPML Lab 5 - Part A
# Uses ResNet + CIFAR-10 setup from your Lab 2 file c2.py

import os
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Import your Lab 2 code
from c2 import BasicBlock, ResNet   # c2.py must be in the same directory


# --------------------------
# Argument parsing
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="HPML Lab 5 - DDP CIFAR10 Training (using Lab 2 model)",
        allow_abbrev=False,
        add_help=True,
    )
    # Match Lab 2 defaults where it makes sense
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to CIFAR-10 dataset (same as Lab 2)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader num_workers (reuse Lab 2 value)")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size PER GPU")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate (Lab 2 default = 0.1)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (Lab 2 default = 0.9)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (Lab 2 default = 5e-4)")

    parser.add_argument("--no_ddp", action="store_true",
                        help="Disable DDP and run on a single GPU only")
    parser.add_argument("--measure_epoch", type=int, default=1,
                        help="Epoch index to measure timing (0-based). "
                             "Use 1 to warm up on epoch 0 and time epoch 1.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


# --------------------------
# Seeding
# --------------------------
def setup_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# DDP setup / cleanup
# --------------------------
def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """
    Initialize torch.distributed for multi-GPU training.
    Assumes launch via `torchrun` so LOCAL_RANK etc. are set.
    """
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# --------------------------
# Model + Data
# --------------------------
def build_model(num_classes: int = 10) -> nn.Module:
    """
    Build the same ResNet18 you used in Lab 2:
      net = ResNet(BasicBlock, [2, 2, 2, 2])
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def build_dataloaders(
    data_path: str,
    batch_size: int,
    num_workers: int,
    use_ddp: bool,
    world_size: int,
    rank: int,
):
    """
    CIFAR-10 Dataloaders using the same transforms as Lab 2.
    Training loader uses DDP sampler when use_ddp=True.
    """
    # Train transforms (from your Lab 2 code)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    # Test transforms (for completeness; Lab 5 Part A mostly needs train)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform_train,
    )

    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optional test loader (not strictly required for Lab 5 Part A)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform_test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, train_sampler, test_loader


# --------------------------
# Training one epoch
# --------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    measure_epoch: int,
    train_sampler=None,
) -> Tuple[float, float, float, float]:
    """
    Train for one epoch.

    Returns:
      avg_loss, acc, total_time_sec, compute_time_sec

    - total_time_sec: whole epoch incl. DataLoader (for Q2 speedup, comm+compute)
    - compute_time_sec: only CPU→GPU + forward + backward + update,
      summed over iterations (for Q1, and for separating compute vs comm in Q3.1)
    """
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    measure = (epoch == measure_epoch)
    total_time_sec = 0.0
    compute_time_sec = 0.0

    if measure:
        torch.cuda.synchronize(device)
        t_epoch_start = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Everything below counts as "compute time" when measure=True
        if measure:
            t_iter_start = time.perf_counter()

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if measure:
            torch.cuda.synchronize(device)
            t_iter_end = time.perf_counter()
            compute_time_sec += (t_iter_end - t_iter_start)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if measure:
        torch.cuda.synchronize(device)
        t_epoch_end = time.perf_counter()
        total_time_sec = t_epoch_end - t_epoch_start

    avg_loss = running_loss / total
    acc = correct / total

    return avg_loss, acc, total_time_sec, compute_time_sec


# --------------------------
# Helper: count gradient size (for Q3.2)
# --------------------------
def count_model_grad_bytes(model: nn.Module, dtype_bytes: int = 4) -> int:
    """
    Approximate total bytes of gradients (assuming FP32 => 4 bytes).
    Use this for Q3.2 ring all-reduce bandwidth calc.
    """
    num_elems = 0
    for p in model.parameters():
        num_elems += p.numel()
    return num_elems * dtype_bytes


# --------------------------
# Main
# --------------------------
def main():
    args = parse_args()
    setup_seed(args.seed)

    use_ddp = not args.no_ddp

    if use_ddp:
        rank, world_size, local_rank, device = setup_ddp()
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data loaders
    train_loader, train_sampler, _ = build_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank,
    )

    # Build model (Lab 2 ResNet) and wrap in DDP if needed
    model = build_model(num_classes=10)
    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer & loss (match Lab 2 hyperparams)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # --------- TRAIN LOOP ---------
    for epoch in range(args.epochs):
        avg_loss, acc, total_time, compute_time = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            measure_epoch=args.measure_epoch,
            train_sampler=train_sampler,
        )

        # Only rank 0 logs anything
        if (not use_ddp) or rank == 0:
            print(f"[EPOCH] {epoch} | loss={avg_loss:.4f} | acc={acc:.4f}")

            if epoch == args.measure_epoch:
                print(
                    f"[TIME] epoch={epoch} | "
                    f"total_time_sec={total_time:.6f} | "
                    f"compute_time_sec={compute_time:.6f}"
                )

    # Optional: print gradient size once (for Q3.2)
    if (not use_ddp) or rank == 0:
        # If wrapped in DDP, the real model is model.module
        raw_model = model.module if isinstance(model, DDP) else model
        bytes_grad = count_model_grad_bytes(raw_model, dtype_bytes=4)
        print(f"[MODEL] total gradient size ≈ {bytes_grad / (1024**2):.2f} MB")

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
