#!/usr/bin/env python3
"""
Hardware detection and auto-scaling configuration generator.
Interrogates CUDA, CPU, RAM and generates configs/hardware/auto.yaml.
Supports manual overrides via --overlay <file.yaml>.
"""

import argparse
import os
from pathlib import Path

import psutil
import torch
import yaml


def detect_hardware():
    config = {}

    config['cpu'] = {
        'num_cores': psutil.cpu_count(logical=True),
        'num_physical_cores': psutil.cpu_count(logical=False),
    }

    mem = psutil.virtual_memory()
    config['ram'] = {
        'total_gb': round(mem.total / (1024 ** 3), 1),
        'available_gb': round(mem.available / (1024 ** 3), 1),
    }

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        config['ddp'] = num_gpus > 1
        config['num_gpus'] = num_gpus

        gpus = []
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                'name': props.name,
                'vram_gb': round(props.total_memory / (1024 ** 3), 1),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count,
            })

        config['gpus'] = gpus

        primary_gpu = gpus[0]
        cap_major = int(primary_gpu['compute_capability'].split('.')[0])
        supports_bf16 = cap_major >= 8
        config['precision'] = 'bf16' if supports_bf16 else 'fp16'

        total_vram = sum(g['vram_gb'] for g in gpus) / len(gpus)
        micro_batch = max(1, int(total_vram * 0.4))
        micro_batch = min(micro_batch, 16)
        effective_target = micro_batch * 1024 * 16
        config['batch'] = {
            'micro_batch_size': micro_batch,
            'target_effective_tokens_per_step': effective_target,
        }

        config['dataloader'] = {
            'num_workers': min(psutil.cpu_count(logical=False) or 4, 8),
            'pin_memory': True,
            'prefetch_factor': 2,
        }

        config['gradient_checkpointing'] = total_vram < 32
        config['allow_tf32'] = cap_major >= 8
        config['use_torch_compile'] = True
        config['compile_mode'] = 'default'
    else:
        config['ddp'] = False
        config['num_gpus'] = 0
        config['precision'] = 'fp32'
        config['batch'] = {'micro_batch_size': 1, 'target_effective_tokens_per_step': 524288}
        config['dataloader'] = {'num_workers': 0, 'pin_memory': False, 'prefetch_factor': 1}
        config['gradient_checkpointing'] = False
        config['allow_tf32'] = False
        config['use_torch_compile'] = False

    return config


def load_overlay(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def deep_merge(base, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def main():
    parser = argparse.ArgumentParser(description='Hardware detection and config generation')
    parser.add_argument('--overlay', type=str, default=None,
                        help='YAML file with manual overrides for the auto-generated config')
    parser.add_argument('--output', type=str,
                        default='configs/hardware/auto.yaml',
                        help='Output path for generated config')
    args = parser.parse_args()

    config = detect_hardware()

    if args.overlay:
        overlay = load_overlay(args.overlay)
        config = deep_merge(config, overlay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Hardware config written to {output_path}")
    print(f"  GPUs: {config.get('num_gpus', 0)}")
    print(f"  Precision: {config.get('precision', 'fp32')}")
    print(f"  Micro batch: {config['batch']['micro_batch_size']}")
    print(f"  DDP: {config.get('ddp', False)}")
    print(f"  Gradient checkpointing: {config.get('gradient_checkpointing', False)}")
    print(f"  Torch compile: {config.get('use_torch_compile', False)}")


if __name__ == '__main__':
    main()
