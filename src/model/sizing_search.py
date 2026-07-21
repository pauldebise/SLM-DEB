#!/usr/bin/env python3
"""
Search for model dimensions (n_layers, d_model, n_heads, d_ff) that reach
a target parameter count within ±3%. Uses a grid search and outputs YAML
config files for each target.

Usage:
  python src/model/sizing_search.py                    # default targets: 100M, 300M, 800M
  python src/model/sizing_search.py --targets 50M,200M  # custom targets
"""

import argparse
import math
import os
from pathlib import Path

import yaml


def param_count(n_layers: int, d_model: int, d_ff: int, vocab_size: int) -> int:
    emb = vocab_size * d_model
    per_layer = 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model
    final_norm = d_model
    return emb + n_layers * per_layer + final_norm


def round_ff(d_model: int, multiple: int = 64) -> int:
    raw = int((8 / 3) * d_model)
    return ((raw + multiple - 1) // multiple) * multiple


def grid_search(target: int, vocab_size: int, tolerance: float = 0.03):
    best_config = None
    best_error = float("inf")

    for n_layers in range(4, 48):
        for d_model in range(128, 2048, 64):
            for n_heads in range(4, 33):
                if d_model % n_heads != 0:
                    continue
                head_dim = d_model // n_heads
                if head_dim < 32 or head_dim > 256:
                    continue

                d_ff = round_ff(d_model)

                params = param_count(n_layers, d_model, d_ff, vocab_size)
                error = abs(params - target) / target

                if error < tolerance and error < best_error:
                    best_error = error
                    best_config = {
                        "n_layers": n_layers,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "d_ff": d_ff,
                        "head_dim": head_dim,
                        "params": params,
                        "error_pct": round(error * 100, 2),
                    }

    return best_config


def generate_config(target_name: str, target_params: int, config: dict,
                    vocab_size: int, max_seq_len: int, dropout: float,
                    output_dir: str):
    cfg = {
        "model": {
            "name": f"slm-{target_name}",
            "description": f"SLM {target_name} ({target_params // 1_000_000}M parameters target)",
            "vocab_size": vocab_size,
            "d_model": config["d_model"],
            "n_layers": config["n_layers"],
            "n_heads": config["n_heads"],
            "d_ff": config["d_ff"],
            "head_dim": config["head_dim"],
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "theta": 10000.0,
            "estimated_params": config["params"],
            "error_pct": config["error_pct"],
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{target_name}.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  Wrote {path}")
    print(f"    n_layers={config['n_layers']}, d_model={config['d_model']}, "
          f"n_heads={config['n_heads']}, d_ff={config['d_ff']}")
    print(f"    Estimated params: {config['params']:,} ({config['error_pct']:+.2f}% error)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Search model dimensions for target param counts")
    parser.add_argument("--targets", type=str, default="100M,300M,800M",
                        help="Comma-separated target sizes (e.g., 100M,300M,800M)")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--output-dir", type=str, default="configs/model")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()

    targets = {}
    for t_str in args.targets.split(","):
        t_str = t_str.strip().upper()
        if t_str.endswith("M"):
            targets[t_str.lower()] = int(t_str[:-1]) * 1_000_000
        elif t_str.endswith("B"):
            targets[t_str.lower()] = int(t_str[:-1]) * 1_000_000_000
        else:
            try:
                targets[t_str.lower()] = int(t_str)
            except ValueError:
                print(f"Invalid target: {t_str}")

    print(f"Vocab size: {args.vocab_size}, max_seq_len: {args.max_seq_len}")
    print(f"Tolerance: {args.tolerance*100}%")
    print(f"Targets: {list(targets.keys())}\n")

    for name, target in targets.items():
        print(f"=== Searching for {name} ({target:,} params) ===")
        config = grid_search(target, args.vocab_size, args.tolerance)
        if config is None:
            print(f"  No config found within tolerance!")
            continue
        generate_config(name, target, config, args.vocab_size,
                        args.max_seq_len, args.dropout, args.output_dir)

    print("\nDone. Run with: python src/model/sizing_search.py")


if __name__ == "__main__":
    main()
