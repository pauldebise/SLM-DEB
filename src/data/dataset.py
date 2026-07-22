#!/usr/bin/env python3
"""
PyTorch IterableDataset for pre-tokenized binary shards.
Memory-maps shard files, supports shard shuffling, and yields
fixed-length token sequences for training.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset


class SLMDataset(IterableDataset):
    def __init__(self, manifest_path: str, seq_len: int, split: str = "train",
                 shuffle_shards: bool = True, seed: int = 42,
                 rank: int = 0, world_size: int = 1):
        self._manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        with open(manifest_path) as f:
            manifest = json.load(f)

        key = "train_shards" if split == "train" else "val_shards"
        self.shards = manifest[key]
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        if not self.shards:
            raise ValueError(f"No shards found for split '{split}' in manifest")

        total_tokens = sum(s["num_tokens"] for s in self.shards)
        num_sequences = total_tokens // seq_len
        num_sequences_per_worker = num_sequences // world_size
        self._len = num_sequences_per_worker

    def __len__(self):
        return self._len

    def _load_shard(self, shard_info):
        path = shard_info["path"]
        if not os.path.isabs(path):
            path = os.path.join(self._manifest_dir, os.path.basename(path))
        return np.memmap(path, dtype=np.uint16, mode="r")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            seed = self.seed + worker_id
        else:
            worker_id = 0
            num_workers = 1
            seed = self.seed

        rng = random.Random(seed)

        shard_order = list(range(len(self.shards)))
        if self.shuffle_shards:
            rng.shuffle(shard_order)

        carryover = np.array([], dtype=np.uint16)

        for shard_idx in shard_order:
            if shard_idx % num_workers != worker_id:
                continue

            shard_info = self.shards[shard_idx]
            data = self._load_shard(shard_info)

            if len(carryover) > 0:
                data = np.concatenate([carryover, data])

            num_tokens = len(data)
            num_full = (num_tokens // self.seq_len) * self.seq_len

            for i in range(0, num_full, self.seq_len):
                chunk = data[i:i + self.seq_len]
                yield {
                    "input_ids": torch.from_numpy(chunk.astype(np.int64)),
                    "labels": torch.from_numpy(chunk.astype(np.int64)),
                }

            remainder_start = num_full
            if remainder_start < num_tokens:
                carryover = data[remainder_start:].copy()
            else:
                carryover = np.array([], dtype=np.uint16)


def build_dataloader(manifest_path: str, seq_len: int, micro_batch_size: int,
                     split: str = "train", num_workers: int = 0,
                     shuffle: bool = True, seed: int = 42,
                     rank: int = 0, world_size: int = 1,
                     prefetch_factor: int = 2,
                     persistent_workers: bool = True,
                     pin_memory: bool = True):
    dataset = SLMDataset(
        manifest_path=manifest_path,
        seq_len=seq_len,
        split=split,
        shuffle_shards=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the SLM dataset loader")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=5)
    args = parser.parse_args()

    ds = SLMDataset(
        manifest_path=args.manifest,
        seq_len=args.seq_len,
        split="train",
        shuffle_shards=False,
    )

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=0,
    )

    for i, batch in enumerate(loader):
        print(f"Batch {i}: input_ids shape={batch['input_ids'].shape}, "
              f"labels shape={batch['labels'].shape}")
        print(f"  First 10 tokens: {batch['input_ids'][0, :10].tolist()}")
        if i >= args.num_batches - 1:
            break

    print(f"Done. Dataset size: {len(ds)} sequences")
