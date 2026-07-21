#!/usr/bin/env python3
"""
Evaluation script for SLM checkpoints.
Supports perplexity evaluation on validation data and basic text generation
for qualitative assessment.

Usage:
  python src/eval.py --checkpoint checkpoints/checkpoint_step_0001000.pt --manifest data/shards/manifest.json
  python src/eval.py --checkpoint checkpoints/checkpoint_step_0001000.pt --prompt "Once upon a time"
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import yaml

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from model.transformer import Transformer, TransformerConfig
from data.dataset import build_dataloader


def _detect_split(manifest_path):
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("val_shards"):
        return "val"
    return "train"


def evaluate_perplexity(model, device, manifest_path, seq_len, batch_size, max_batches=100):
    split = _detect_split(manifest_path)
    loader = build_dataloader(
        manifest_path=manifest_path,
        seq_len=seq_len,
        micro_batch_size=batch_size,
        split=split,
        num_workers=2,
        shuffle=False,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels)
            total_loss += out["loss"].item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def generate_text(model, device, tokenizer, prompt, max_new_tokens=200,
                  temperature=0.8, top_k=50, top_p=0.9):
    model.eval()
    if tokenizer:
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    else:
        input_ids = torch.tensor(
            [[hash(c) % 32000 + 1 for c in prompt]],
            dtype=torch.long, device=device,
        )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

    new_ids = output_ids[0, input_ids.size(1):].tolist()
    if tokenizer:
        return tokenizer.decode(new_ids, skip_special_tokens=False)
    return " ".join(str(t) for t in new_ids)


def main():
    parser = argparse.ArgumentParser(description="SLM evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_dict = ckpt.get("config", {})
    state_dict = ckpt.get("model_state_dict", {})

    tokenizer = None
    if os.path.exists(args.tokenizer):
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(args.tokenizer)
        except Exception as e:
            print(f"Could not load tokenizer: {e}")

    tcfg = TransformerConfig(config_dict)
    model = Transformer(tcfg).to(device)
    model.load_state_dict(state_dict)

    n_params = sum(p.numel() for p in model.parameters())
    step = ckpt.get("step", "unknown")
    print(f"Checkpoint: step={step}, params={n_params/1e6:.1f}M")

    if args.manifest:
        loss, ppl = evaluate_perplexity(
            model, device, args.manifest,
            config_dict["max_seq_len"], args.batch_size, args.max_batches,
        )
        print(f"Val loss: {loss:.4f}, perplexity: {ppl:.2f}")

    if args.prompt:
        generated = generate_text(
            model, device, tokenizer, args.prompt, max_new_tokens=200,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
