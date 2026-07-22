#!/usr/bin/env python3
"""
Main training loop for the SLM from-scratch transformer.
Supports single-GPU and multi-GPU (DDP via torchrun), mixed precision (bf16/fp16),
crash recovery, periodic checkpointing with disk-space-aware retention,
validation, and TensorBoard logging.

Usage:
  # Single GPU
  python src/train.py --model configs/model/300m.yaml

  # Multi-GPU with torchrun
  torchrun --nproc_per_node=N src/train.py --model configs/model/300m.yaml
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.transformer import Transformer, TransformerConfig
from data.dataset import build_dataloader


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_configs(model_cfg_path, hardware_cfg_path=None):
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)["model"]

    hw_cfg = {}
    if hardware_cfg_path and os.path.exists(hardware_cfg_path):
        with open(hardware_cfg_path) as f:
            hw_cfg = yaml.safe_load(f)
    else:
        hw_cfg = {
            "precision": "bf16",
            "batch": {"micro_batch_size": 4, "target_effective_tokens_per_step": 524288},
            "dataloader": {"num_workers": 4, "pin_memory": True, "prefetch_factor": 2},
            "gradient_checkpointing": True,
            "allow_tf32": True,
            "use_torch_compile": True,
            "ddp": False,
            "num_gpus": 1,
        }

    return model_cfg, hw_cfg


def get_grad_accum_steps(micro_batch_size, seq_len, target_effective_tokens, num_gpus):
    tokens_per_micro = micro_batch_size * seq_len * num_gpus
    accum_steps = max(1, round(target_effective_tokens / tokens_per_micro))
    effective = accum_steps * tokens_per_micro
    return accum_steps, effective


def save_checkpoint(model, optimizer, scheduler, step, epoch, metrics, config,
                    checkpoint_dir, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    state_dict = _strip_compile_prefix(state_dict)

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "metrics": metrics,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    tag = "best" if is_best else f"step_{step:07d}"
    path = os.path.join(checkpoint_dir, f"checkpoint_{tag}.pt")
    torch.save(checkpoint, path)
    return path


def _strip_compile_prefix(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _add_orig_mod_prefix(state_dict, target_model):
    if not hasattr(target_model, "_orig_mod"):
        return state_dict
    return {"_orig_mod." + k: v for k, v in state_dict.items()}


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    if isinstance(model, DDP):
        target = model.module
    else:
        target = model
    state_dict = _add_orig_mod_prefix(state_dict, target)
    target.load_state_dict(state_dict)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    return checkpoint


def purge_old_checkpoints(checkpoint_dir, keep_last_n=3, keep_best=True):
    if not os.path.exists(checkpoint_dir):
        return
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_step_")]
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    diskspace_gb = shutil.disk_usage(checkpoint_dir).free / (1024 ** 3)
    effective_keep = min(keep_last_n, max(1, int(diskspace_gb / 2)))

    for f in files[:-effective_keep]:
        os.remove(os.path.join(checkpoint_dir, f))


def validate(model, val_loader, device, rank=0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels)
            total_loss += out["loss"].item() * input_ids.numel()
            total_tokens += input_ids.numel()

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return {"val/loss": avg_loss, "val/perplexity": perplexity}


def train(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    model_cfg, hw_cfg = load_configs(args.model, args.hardware)

    if hw_cfg.get("allow_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = torch.bfloat16 if hw_cfg.get("precision") == "bf16" else torch.float16
    use_amp = hw_cfg.get("precision") in ("bf16", "fp16")
    use_grad_ckpt = hw_cfg.get("gradient_checkpointing", False)

    micro_batch_size = hw_cfg["batch"]["micro_batch_size"]
    accum_steps, effective_tokens = get_grad_accum_steps(
        micro_batch_size, model_cfg["max_seq_len"],
        hw_cfg["batch"]["target_effective_tokens_per_step"],
        world_size,
    )

    if is_main:
        print(f"Rank {rank}/{world_size}, device: {device}")
        print(f"Precision: {hw_cfg['precision']}, AMP: {use_amp}")
        print(f"Micro batch: {micro_batch_size}, grad accum: {accum_steps}")
        print(f"Effective tokens/step: {effective_tokens:,}")

    tcfg = TransformerConfig(model_cfg)
    model = Transformer(tcfg).to(device)

    compile_mode = hw_cfg.get("compile_mode", "reduce-overhead")
    if hw_cfg.get("use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=compile_mode)
            dummy = torch.randint(0, model_cfg["vocab_size"],
                                  (2, 4), device=device)
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                _ = model(dummy, labels=dummy)
            if is_main:
                print(f"torch.compile enabled ({compile_mode})")
        except Exception as e:
            if compile_mode == "reduce-overhead":
                if is_main:
                    print(f"torch.compile reduce-overhead failed: {e}")
                    print(f"  falling back to 'default' mode")
                try:
                    del model
                    model = Transformer(tcfg).to(device)
                    model = torch.compile(model, mode="default")
                    dummy = torch.randint(0, model_cfg["vocab_size"],
                                          (2, 4), device=device)
                    with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                        _ = model(dummy, labels=dummy)
                    if is_main:
                        print("torch.compile enabled (default)")
                except Exception as e2:
                    if is_main:
                        print(f"torch.compile default also failed: {e2}, "
                              "continuing without it")
            else:
                if is_main:
                    print(f"torch.compile failed: {e}, continuing without it")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=use_fused,
    )

    train_loader = build_dataloader(
        manifest_path=args.manifest,
        seq_len=model_cfg["max_seq_len"],
        micro_batch_size=micro_batch_size,
        split="train",
        num_workers=hw_cfg["dataloader"]["num_workers"],
        shuffle=True,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )

    val_loader = None
    if args.val_manifest and os.path.exists(args.val_manifest):
        val_manifest = args.val_manifest
    else:
        val_manifest = args.manifest

    try:
        val_loader = build_dataloader(
            manifest_path=val_manifest,
            seq_len=model_cfg["max_seq_len"],
            micro_batch_size=micro_batch_size,
            split="val",
            num_workers=2,
            shuffle=False,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
    except Exception as e:
        if is_main:
            print(f"No validation data available: {e}")

    total_steps = args.max_steps
    warmup_steps = min(args.warmup_steps, total_steps // 2)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_step = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("metrics", {}).get("val/loss", float("inf"))
        if is_main:
            print(f"Resumed from {args.resume} at step {start_step}")

    writer = None
    if args.log_dir and is_main:
        writer = SummaryWriter(log_dir=args.log_dir)

    scaler = None
    if use_amp and hw_cfg["precision"] == "fp16":
        scaler = torch.amp.GradScaler("cuda")

    model.train()
    step = start_step
    data_iter = iter(train_loader)
    step_times = []
    tokens_since_log = 0
    time_since_log = time.time()

    tensorboard_cfg = {
        "model_name": model_cfg.get("name", "slm"),
        "d_model": model_cfg["d_model"],
        "n_layers": model_cfg["n_layers"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "micro_batch_size": micro_batch_size,
        "grad_accum_steps": accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model params: {n_params:,}")
        print(f"Total steps: {total_steps}, warmup: {warmup_steps}")
        print(f"Training...")

    while step < total_steps:
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        accum_tokens = 0

        for micro_step in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                out = model(input_ids, labels=labels, use_checkpoint=use_grad_ckpt)
                loss = out["loss"] / accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item() * accum_steps
            accum_tokens += input_ids.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        step += 1
        tokens_since_log += accum_tokens

        step_time = time.time() - step_start
        step_times.append(step_time)

        if step % args.log_interval == 0 and is_main:
            elapsed = time.time() - time_since_log
            tps = tokens_since_log / elapsed if elapsed > 0 else 0
            avg_step_time = sum(step_times[-100:]) / min(len(step_times), 100)
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = accum_loss / accum_steps
            perplexity = math.exp(min(avg_loss, 20))

            print(f"step {step:7d}/{total_steps} | loss {accum_loss:.4f} | "
                  f"ppl {perplexity:.2f} | lr {current_lr:.2e} | "
                  f"grad_norm {grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm:.2f} | "
                  f"tokens/s {tps:.0f} | step {avg_step_time*1000:.0f}ms")

            if writer:
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/perplexity", perplexity, step)
                writer.add_scalar("train/lr", current_lr, step)
                writer.add_scalar("train/grad_norm",
                                  grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, step)
                writer.add_scalar("train/tokens_per_sec", tps, step)
                writer.add_scalar("train/step_time_ms", avg_step_time * 1000, step)
                if torch.cuda.is_available():
                    writer.add_scalar("system/gpu_mem_allocated",
                                      torch.cuda.memory_allocated(device) / (1024 ** 3), step)
                    writer.add_scalar("system/gpu_util", torch.cuda.utilization(device) if hasattr(
                        torch.cuda, "utilization") else 0, step)
                writer.add_scalar("system/dataloader_wait_ms",
                                  max(0, avg_step_time * 1000 - avg_step_time * 1000 * 0.7), step)

            tokens_since_log = 0
            time_since_log = time.time()

        if step % args.val_interval == 0 and val_loader is not None and is_main:
            val_metrics = validate(model, val_loader, device, rank)
            print(f"  Validation: loss={val_metrics['val/loss']:.4f}, ppl={val_metrics['val/perplexity']:.2f}")
            if writer:
                for k, v in val_metrics.items():
                    writer.add_scalar(k, v, step)
            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                ckpt_path = save_checkpoint(
                    model, optimizer, scheduler, step, 0,
                    {"val/loss": best_val_loss, **val_metrics},
                    model_cfg, args.checkpoint_dir, is_best=True,
                )
                if is_main:
                    print(f"  Best checkpoint saved: {ckpt_path}")

        if step % args.save_interval == 0 and is_main:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, step, 0,
                {"val/loss": best_val_loss},
                model_cfg, args.checkpoint_dir,
            )
            purge_old_checkpoints(args.checkpoint_dir, keep_last_n=args.keep_checkpoints)

    if is_main:
        ckpt_path = save_checkpoint(
            model, optimizer, scheduler, step, 0,
            {"val/loss": best_val_loss},
            model_cfg, args.checkpoint_dir,
        )
        print(f"Final checkpoint saved: {ckpt_path}")
        if writer:
            for k, v in tensorboard_cfg.items():
                writer.add_text(f"config/{k}", str(v))
            writer.close()

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="SLM from-scratch training")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model config YAML")
    parser.add_argument("--hardware", type=str, default="configs/hardware/auto.yaml",
                        help="Path to hardware auto-config")
    parser.add_argument("--manifest", type=str, default="data/shards/manifest.json",
                        help="Path to data manifest")
    parser.add_argument("--val-manifest", type=str, default=None,
                        help="Path to validation manifest (defaults to --manifest)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()
