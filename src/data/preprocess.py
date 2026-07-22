#!/usr/bin/env python3
"""
Pre-tokenize datasets into binary memory-mapped shards.
Streams from HuggingFace datasets, tokenizes with our BPE tokenizer,
and writes uint16 numpy arrays. Produces separate train/val shards.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

_log_file = None


def _progress_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True, **kwargs)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


def format_chat(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        elif role == "system":
            parts.append(f"<|system|>\n{content}")
    return "\n".join(parts)


def check_disk_space(output_dir, estimated_bytes):
    usage = shutil.disk_usage(output_dir if os.path.exists(output_dir) else "/workspace")
    free_gb = usage.free / (1024 ** 3)
    needed_gb = estimated_bytes / (1024 ** 3)
    _progress_print(f"  Disk free: {free_gb:.1f} GB, estimated need: {needed_gb:.1f} GB")
    if usage.free < estimated_bytes * 1.5:
        _progress_print(f"  WARNING: less than 50% headroom after estimated writes")
    return usage.free >= estimated_bytes


def preprocess_source(tokenizer, source_config, output_dir, shard_size, val_fraction, rng,
                      max_tokens=None, source_name="unknown", global_shard_start=0,
                      total_target_tokens=None):
    dataset_id = source_config["dataset"]
    ds_config = source_config.get("config")
    split = source_config.get("split", "train")
    field = source_config.get("field", "text")
    filter_lang = source_config.get("filter_lang")

    load_kwargs = {"path": dataset_id, "split": split, "streaming": True}
    if ds_config:
        load_kwargs["name"] = ds_config

    ds = load_dataset(**load_kwargs)

    train_shards = []
    val_shards = []
    current_buffer = []
    current_val_buffer = []
    token_count = 0
    val_token_count = 0
    sample_count = 0
    shard_idx = global_shard_start

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    def flush_buffer(buffer, is_val):
        nonlocal shard_idx
        if not buffer:
            return None
        flat = np.concatenate(buffer).astype(np.uint16)
        prefix = "val" if is_val else "train"
        path = os.path.join(output_dir, f"{source_name}_{prefix}_shard_{shard_idx:04d}.bin")
        flat.tofile(path)
        meta = {
            "path": path,
            "num_tokens": len(flat),
            "shard_idx": shard_idx,
            "source": source_name,
        }
        shard_idx += 1
        _progress_print(f"    Wrote {prefix} shard: {len(flat):,} tokens -> {path}")
        return meta

    for example in ds:
        if filter_lang and example.get("lang") != filter_lang:
            continue

        if field == "messages":
            text = format_chat(example.get(field, []))
        else:
            text = example.get(field, "")

        if not text or len(text.strip()) < 20:
            continue

        encoded = tokenizer.encode(text, add_special_tokens=False)
        ids = encoded.ids
        if not ids:
            continue

        ids_with_special = [bos_id] + ids + [eos_id]
        ids_arr = np.array(ids_with_special, dtype=np.uint16)

        is_val = rng.random() < val_fraction

        if is_val:
            current_val_buffer.append(ids_arr)
            val_token_count += len(ids_arr)
            total_buffer = sum(len(b) for b in current_val_buffer)
            if total_buffer >= shard_size:
                meta = flush_buffer(current_val_buffer, is_val=True)
                if meta:
                    val_shards.append(meta)
                current_val_buffer = []
        else:
            current_buffer.append(ids_arr)
            token_count += len(ids_arr)
            total_buffer = sum(len(b) for b in current_buffer)
            if total_buffer >= shard_size:
                meta = flush_buffer(current_buffer, is_val=False)
                if meta:
                    train_shards.append(meta)
                current_buffer = []

        sample_count += 1
        if sample_count % 10000 == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (token_count + val_token_count) / elapsed if elapsed > 0 else 0
            eta_str = ""
            if max_tokens and (token_count + val_token_count) > 0:
                remaining = max_tokens - (token_count + val_token_count)
                eta_sec = remaining / tok_per_sec if tok_per_sec > 0 else 0
                eta_str = f"ETA {eta_sec/60:.0f}m " if eta_sec > 0 else ""
            _progress_print(f"  [{source_name}] {sample_count:,} samples, {token_count + val_token_count:,} tokens "
                  f"({tok_per_sec/1e6:.1f}M tok/s) {eta_str}"
                  f"| train={token_count/1e6:.1f}M val={val_token_count/1e6:.1f}M | "
                  f"shards={shard_idx - global_shard_start}")

        if max_tokens and token_count + val_token_count >= max_tokens:
            break

    meta = flush_buffer(current_buffer, is_val=False)
    if meta:
        train_shards.append(meta)
    meta = flush_buffer(current_val_buffer, is_val=True)
    if meta:
        val_shards.append(meta)

    return train_shards, val_shards, token_count, val_token_count


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize datasets into binary shards")
    parser.add_argument("--config", type=str, default="configs/data/mixture.yaml")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="data/shards")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum tokens to produce (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to a progress log file (in addition to stdout)")
    args = parser.parse_args()

    global _log_file
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        _log_file = open(args.log_file, "a", buffering=1)

    if not os.path.exists(args.tokenizer):
        _progress_print(f"Tokenizer not found at {args.tokenizer}. Train it first with src/tokenizer/train_tokenizer.py")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    _progress_print(f"Loaded tokenizer: vocab_size={tokenizer.get_vocab_size()}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    shard_size = cfg["shard"]["tokens_per_shard"]
    val_fraction = cfg["validation"]["fraction"]
    sources = cfg["sources"]

    _progress_print(f"Shard size: {shard_size:,} tokens, val fraction: {val_fraction}")

    rng = np.random.RandomState(args.seed)

    start_time = time.time()

    all_train_shards = []
    all_val_shards = []
    total_train_tokens = 0
    total_val_tokens = 0
    global_shard = 0

    for name, src_cfg in sources.items():
        _progress_print(f"\n=== Processing source: {name} ===")
        weight = src_cfg.get("weight", 1)
        total_weight = sum(s["weight"] for s in sources.values())
        frac = weight / total_weight
        src_max_tokens = int(args.max_tokens * frac) if args.max_tokens else None
        if src_max_tokens is not None:
            _progress_print(f"  Target max tokens: {src_max_tokens:,}")

        try:
            train_shards, val_shards, n_train, n_val = preprocess_source(
                tokenizer, src_cfg, args.output_dir, shard_size, val_fraction, rng,
                max_tokens=src_max_tokens, source_name=name,
                global_shard_start=global_shard
            )
            global_shard += len(train_shards) + len(val_shards)
            all_train_shards.extend(train_shards)
            all_val_shards.extend(val_shards)
            total_train_tokens += n_train
            total_val_tokens += n_val
            _progress_print(f"  Source '{name}' done: {n_train:,} train tokens, {n_val:,} val tokens")
        except Exception as e:
            _progress_print(f"  ERROR processing source '{name}': {e}")
            _progress_print(f"  Skipping '{name}' and continuing with remaining sources.")
            import traceback
            _progress_print(traceback.format_exc())

    # Write manifest
    manifest = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.get_vocab_size(),
        "shard_size": shard_size,
        "dtype": "uint16",
        "total_train_tokens": total_train_tokens,
        "total_val_tokens": total_val_tokens,
        "val_fraction": val_fraction,
        "train_shards": all_train_shards,
        "val_shards": all_val_shards,
    }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    _progress_print(f"\n=== Done ===")
    _progress_print(f"  Train tokens: {total_train_tokens:,}")
    _progress_print(f"  Val tokens: {total_val_tokens:,}")
    _progress_print(f"  Train shards: {len(all_train_shards)}")
    _progress_print(f"  Val shards: {len(all_val_shards)}")

    total_bytes = (total_train_tokens + total_val_tokens) * 2
    elapsed = time.time() - start_time
    _progress_print(f"  Total size: {total_bytes / (1024**3):.2f} GB")
    _progress_print(f"  Elapsed: {elapsed/60:.1f} min")
    if elapsed > 0:
        _progress_print(f"  Throughput: {(total_train_tokens + total_val_tokens) / elapsed / 1e6:.1f} M tok/s")
    check_disk_space(args.output_dir, total_bytes * 1.2)

    if _log_file is not None:
        _log_file.close()

    if total_train_tokens == 0:
        _progress_print("ERROR: No training tokens produced. Aborting.")
        os._exit(1)

    os._exit(0)


if __name__ == "__main__":
    main()
