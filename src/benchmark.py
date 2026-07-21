#!/usr/bin/env python3
"""
Micro-benchmark for the SLM training pipeline.
Runs a fixed number of steps with real data and measures:
  - tokens/sec, step time, GPU memory, MFU, dataloader wait time.

Usage:
  python src/benchmark.py --model configs/model/300m.yaml --steps 200
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.transformer import Transformer, TransformerConfig
from data.dataset import build_dataloader


def estimate_flops_per_token(d_model: int, n_layers: int, d_ff: int) -> int:
    attn_flops = 4 * d_model * d_model + 2 * d_model * d_model
    mlp_flops = 3 * d_model * d_ff * 2
    flops_per_layer = attn_flops + mlp_flops
    return n_layers * flops_per_layer


def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.model) as f:
        model_cfg = yaml.safe_load(f)["model"]

    hw_cfg = {
        "precision": "bf16",
        "batch": {"micro_batch_size": args.batch_size, "target_effective_tokens_per_step": args.batch_size * model_cfg["max_seq_len"]},
        "dataloader": {"num_workers": args.num_workers, "pin_memory": True, "prefetch_factor": 2},
        "gradient_checkpointing": args.gradient_checkpointing,
        "allow_tf32": True,
        "use_torch_compile": args.torch_compile,
        "ddp": False,
        "num_gpus": 1,
    }

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tcfg = TransformerConfig(model_cfg)
    model = Transformer(tcfg).to(device)

    if args.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}. Generate data first.")
        sys.exit(1)

    loader = build_dataloader(
        manifest_path=args.manifest,
        seq_len=model_cfg["max_seq_len"],
        micro_batch_size=args.batch_size,
        split="train",
        num_workers=args.num_workers,
        shuffle=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = estimate_flops_per_token(model_cfg["d_model"], model_cfg["n_layers"], model_cfg["d_ff"])
    peak_bf16_tflops = 82.6  # RTX 4090

    print(f"Model: {n_params/1e6:.1f}M params, {model_cfg['n_layers']} layers, d_model={model_cfg['d_model']}")
    print(f"Batch size: {args.batch_size}, seq_len: {model_cfg['max_seq_len']}")
    print(f"FLOPs/token: {flops_per_token/1e6:.1f}M")
    print(f"Warmup steps: {args.warmup_steps}, benchmark steps: {args.steps}")
    print()

    dtype = torch.bfloat16
    data_iter = iter(loader)
    model.train()

    step_times = []
    gpu_mem_peaks = []
    tokens_total = 0
    dataloader_wait_total = 0.0

    for step in range(args.warmup_steps + args.steps):
        dt_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        dt_wait = time.time() - dt_start

        step_start = time.time()

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=dtype):
            out = model(input_ids, labels=labels)
            loss = out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        torch.cuda.synchronize()
        step_time = time.time() - step_start

        if step >= args.warmup_steps:
            step_times.append(step_time)
            gpu_mem_peaks.append(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
            tokens_total += input_ids.numel()
            dataloader_wait_total += dt_wait

    torch.cuda.reset_peak_memory_stats(device)

    avg_step_time = sum(step_times) / len(step_times)
    avg_dt_wait = dataloader_wait_total / len(step_times)
    tps = tokens_total / sum(step_times)
    peak_mem = max(gpu_mem_peaks)

    theoretical_max_tps = peak_bf16_tflops * 1e12 / flops_per_token
    actual_flops_per_sec = tps * flops_per_token
    mfu = actual_flops_per_sec / (peak_bf16_tflops * 1e12) * 100

    print(f"=== Results ({len(step_times)} steps) ===")
    print(f"Tokens/sec: {tps:,.0f}")
    print(f"Avg step time: {avg_step_time*1000:.1f} ms")
    print(f"Avg dataloader wait: {avg_dt_wait*1000:.1f} ms ({avg_dt_wait/avg_step_time*100:.1f}%)")
    print(f"Peak GPU memory: {peak_mem:.2f} GB / {torch.cuda.get_device_properties(device).total_memory/(1024**3):.1f} GB")
    print(f"MFU: {mfu:.2f}%")
    print(f"GPU compute time: {(avg_step_time - avg_dt_wait)*1000:.1f} ms ({(1 - avg_dt_wait/avg_step_time)*100:.1f}%)")

    return {
        "tokens_per_sec": tps,
        "step_time_ms": avg_step_time * 1000,
        "dataloader_wait_ms": avg_dt_wait * 1000,
        "peak_gpu_mem_gb": peak_mem,
        "mfu_pct": mfu,
    }


def main():
    parser = argparse.ArgumentParser(description="SLM training micro-benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="data/shards/manifest.json")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
