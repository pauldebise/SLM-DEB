#!/usr/bin/env python3
"""
Download and cache datasets for the SLM training pipeline.
Verifies dataset availability, streams a few samples to warm the HF cache,
and reports expected token counts.
"""

import argparse
import sys

import yaml
from datasets import load_dataset, get_dataset_config_names


def check_and_cache(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sources = config["sources"]
    total_tokens = 0

    for name, src in sources.items():
        dataset_id = src["dataset"]
        ds_config = src.get("config")
        split = src.get("split", "train")
        field = src.get("field", "text")

        print(f"\n=== {name}: {dataset_id} ===")

        try:
            configs = get_dataset_config_names(dataset_id)
            print(f"  Available configs: {configs[:5]}{'...' if len(configs)>5 else ''}")
            if ds_config and ds_config not in configs:
                print(f"  WARNING: config '{ds_config}' not found. Available: {configs}")
        except Exception as e:
            print(f"  Could not list configs: {e}")

        load_kwargs = {"path": dataset_id, "split": split, "streaming": True}
        if ds_config:
            load_kwargs["name"] = ds_config

        try:
            ds = load_dataset(**load_kwargs)
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            continue

        count = 0
        sample_text = ""
        filter_lang = src.get("filter_lang")
        total_chars = 0
        estimated_tokens = 0
        max_estimate = 10_000

        for example in ds:
            if filter_lang and example.get("lang") != filter_lang:
                continue
            if count >= max_estimate:
                break
            f = src.get("field", "text")
            if f == "messages":
                text = _format_chat_estimate(example.get(f, []))
            else:
                text = example.get(f, "")
            if text and len(text.strip()) > 20:
                count += 1
                total_chars += len(text)
                if not sample_text:
                    sample_text = text

        if count > 0:
            avg_chars = total_chars / count
            avg_tokens_per_char = 0.25
            estimated_tokens = int(count * avg_chars * avg_tokens_per_char)
            print(f"  Streamed {count} samples (est. {estimated_tokens:,} tokens)")
            print(f"  Avg chars/sample: {avg_chars:.0f}")
            print(f"  Sample: {sample_text[:100]}...")
        else:
            print(f"  WARNING: no valid samples found")

        total_tokens += estimated_tokens

    print(f"\nTotal estimated tokens (from ~{max_estimate} samples per source): {total_tokens:,}")
    print("Datasets cached successfully.")


def _format_chat_estimate(messages):
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


def main():
    parser = argparse.ArgumentParser(description="Download/cache datasets")
    parser.add_argument("--config", type=str, default="configs/data/mixture.yaml")
    args = parser.parse_args()
    check_and_cache(args.config)


if __name__ == "__main__":
    main()
