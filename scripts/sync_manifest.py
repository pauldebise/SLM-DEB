#!/usr/bin/env python3
"""
Regenerate manifest.json from all .bin shard files found in the shard directory.
Useful when the preprocess writes new shards incrementally or after a crash.

Usage: python3 scripts/sync_manifest.py \
    --tokenizer data/tokenizer/tokenizer.json \
    --shard-dir data/shards \
    --output data/shards/manifest.json
"""

import argparse
import json
import os
import re
import sys

import numpy as np

SHARD_PATTERN = re.compile(r"^(\w+)_(train|val)_shard_(\d+)\.bin$")


def discover_shards(shard_dir):
    train_shards = []
    val_shards = []
    total_train_tokens = 0
    total_val_tokens = 0

    if not os.path.isdir(shard_dir):
        print(f"Shard directory not found: {shard_dir}")
        return None, None, 0, 0

    for fname in sorted(os.listdir(shard_dir)):
        m = SHARD_PATTERN.match(fname)
        if not m:
            continue
        source, split, shard_idx_str = m.group(1), m.group(2), m.group(3)
        shard_idx = int(shard_idx_str)
        full_path = os.path.join(shard_dir, fname)

        file_size = os.path.getsize(full_path)
        if file_size == 0:
            print(f"  WARNING: empty shard {fname}, skipping")
            continue

        try:
            arr = np.memmap(full_path, dtype=np.uint16, mode="r")
            num_tokens = len(arr)
        except Exception as e:
            print(f"  WARNING: cannot read {fname}: {e}, skipping")
            continue

        entry = {
            "path": fname,
            "num_tokens": num_tokens,
            "shard_idx": shard_idx,
            "source": source,
        }

        if split == "train":
            train_shards.append(entry)
            total_train_tokens += num_tokens
        else:
            val_shards.append(entry)
            total_val_tokens += num_tokens

    train_shards.sort(key=lambda s: s["shard_idx"])
    val_shards.sort(key=lambda s: s["shard_idx"])

    return train_shards, val_shards, total_train_tokens, total_val_tokens


def main():
    parser = argparse.ArgumentParser(description="Regenerate manifest from shard files")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer.json",
                        help="Path to tokenizer.json")
    parser.add_argument("--shard-dir", type=str, default="data/shards",
                        help="Directory containing .bin shard files")
    parser.add_argument("--output", type=str, default="data/shards/manifest.json",
                        help="Output manifest JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        print(f"ERROR: tokenizer not found at {args.tokenizer}")
        sys.exit(1)

    train_shards, val_shards, total_train, total_val = discover_shards(args.shard_dir)

    if train_shards is None or len(train_shards) == 0:
        print("ERROR: no valid shards found")
        sys.exit(1)

    total_tokens = total_train + total_val
    shard_size = max(s["num_tokens"] for s in train_shards) if train_shards else 0
    val_fraction = total_val / total_tokens if total_tokens > 0 else 0.0

    manifest = {
        "tokenizer": args.tokenizer,
        "vocab_size": 32768,
        "shard_size": int(shard_size),
        "dtype": "uint16",
        "total_train_tokens": total_train,
        "total_val_tokens": total_val,
        "val_fraction": round(val_fraction, 6),
        "train_shards": train_shards,
        "val_shards": val_shards,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {args.output}")
    print(f"  Train shards: {len(train_shards)} ({total_train:,} tokens, {total_train / 1e9:.2f}B)")
    print(f"  Val shards:   {len(val_shards)} ({total_val:,} tokens, {total_val / 1e6:.2f}M)")

    sources_train = {}
    for s in train_shards:
        src = s["source"]
        sources_train[src] = sources_train.get(src, 0) + s["num_tokens"]
    for src, tok in sorted(sources_train.items()):
        print(f"  [{src}] {tok:,} train tokens ({tok / total_train * 100:.1f}%)")


if __name__ == "__main__":
    main()
