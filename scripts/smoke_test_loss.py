#!/usr/bin/env python3
"""
Non-regression test for the label-shift bug in causal LM loss computation.

The bug: loss was computed as cross_entropy(logits[i], labels[i]) at every
position i, rather than the correct shift: logits[:, :-1] predicts labels[:, 1:].
Since the causal mask allows position i to attend to itself, and input_ids == labels
in the dataset, the model learned the trivial identity shortcut → loss → 0,
grad_norm → 0.

This test:
1. Trains a small model on real data for 150 steps
2. Asserts loss > 0.1 at all steps after step 50 (exactly 0 means the bug is back)
3. Asserts grad_norm > 0 at all steps after step 50
4. Asserts loss decreases over time (the model actually learns)

Usage:
  python scripts/smoke_test_loss.py
  python scripts/smoke_test_loss.py --model configs/model/100m.yaml --steps 200
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

_repo_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(0, str(Path(_repo_root) / "src"))
from model.transformer import Transformer, TransformerConfig
from data.dataset import build_dataloader


def detect_hardware_defaults():
    hw = {
        "precision": "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32",
        "batch": {"micro_batch_size": 4, "target_effective_tokens_per_step": 65536},
        "dataloader": {"num_workers": 0, "pin_memory": True, "prefetch_factor": None, "persistent_workers": False},
        "gradient_checkpointing": False,
        "allow_tf32": False,
        "use_torch_compile": False,
        "ddp": False,
        "num_gpus": 1,
    }
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory
        if free_mem > 20 * 1024 ** 3:
            hw["batch"]["micro_batch_size"] = 8
    return hw


def run_test(model_config_path, manifest_path, max_steps=150, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hw_cfg = detect_hardware_defaults()

    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)["model"]

    micro_batch_size = hw_cfg["batch"]["micro_batch_size"]

    tcfg = TransformerConfig(model_cfg)
    model = Transformer(tcfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    train_loader = build_dataloader(
        manifest_path=manifest_path,
        seq_len=model_cfg["max_seq_len"],
        micro_batch_size=micro_batch_size,
        split="train",
        num_workers=0,
        shuffle=True,
        seed=42,
        pin_memory=True,
        persistent_workers=False,
    )

    dtype = torch.bfloat16 if hw_cfg["precision"] == "bf16" else torch.float32
    use_amp = torch.cuda.is_available()

    model.train()
    data_iter = iter(train_loader)

    loss_history = []
    grad_norm_history = []
    zero_loss_detected = False
    zero_grad_detected = False

    print(f"\nTraining {max_steps} steps...")
    t0 = time.time()

    for step in range(max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
            out = model(input_ids, labels=labels)
            loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        gn_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        loss_history.append(loss_val)
        grad_norm_history.append(gn_val)

        if step > 50:
            if loss_val == 0.0:
                zero_loss_detected = True
                print(f"  *** BUG DETECTED at step {step}: loss = 0.0000 ***")
            if gn_val == 0.0:
                zero_grad_detected = True
                print(f"  *** BUG DETECTED at step {step}: grad_norm = 0.0000 ***")

        if step % 10 == 0 or step == max_steps - 1:
            ppl = math.exp(min(loss_val, 20))
            print(f"  step {step:4d} | loss {loss_val:.4f} | ppl {ppl:.2f} | grad_norm {gn_val:.2f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({max_steps / elapsed:.1f} steps/s)")

    losses_after_warmup = loss_history[50:]
    grad_norms_after_warmup = grad_norm_history[50:]

    print("\n--- Results ---")

    assert not zero_loss_detected, (
        f"FAIL: loss was 0.0000 after step 50. The label-shift bug is still present.\n"
        f"Check transformer.py: logits[:, :-1] must predict labels[:, 1:], "
        f"not logits[i] predicting labels[i]."
    )
    print("PASS: Loss never hit exactly 0 after step 50")

    assert not zero_grad_detected, (
        f"FAIL: grad_norm was 0.0000 after step 50. The model is not learning."
    )
    print("PASS: grad_norm never hit exactly 0 after step 50")

    avg_loss_first_10 = sum(loss_history[50:60]) / 10
    avg_loss_last_10 = sum(loss_history[-10:]) / 10
    print(f"  avg loss (steps 50-60): {avg_loss_first_10:.4f}")
    print(f"  avg loss (last 10):     {avg_loss_last_10:.4f}")

    assert avg_loss_last_10 < avg_loss_first_10 - 0.01, (
        f"FAIL: Loss did not decrease (first={avg_loss_first_10:.4f}, last={avg_loss_last_10:.4f}). "
        f"Expected at least 0.01 decrease."
    )
    print("PASS: Loss decreased over training (model is actually learning)")

    assert loss_history[0] > 0.1, (
        f"FAIL: initial loss too low ({loss_history[0]:.4f}). Expected > 0.1 (random model)."
    )
    print("PASS: Initial loss > 0.1 (random model)")

    expected_initial_loss = math.log(model_cfg["vocab_size"])
    print(f"  Expected initial loss (ln(vocab)): {expected_initial_loss:.2f}")
    print(f"  Actual initial loss: {loss_history[0]:.4f}")

    print("\n=== ALL NON-REGRESSION CHECKS PASSED ===")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test: verify loss computation is correct")
    parser.add_argument("--model", type=str, default="configs/model/300m.yaml")
    parser.add_argument("--manifest", type=str, default="data/shards/manifest.json")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print("Run preprocess first: python src/data/preprocess.py --max-tokens 10000000", file=sys.stderr)
        sys.exit(1)

    ok = run_test(args.model, args.manifest, args.steps, args.lr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
