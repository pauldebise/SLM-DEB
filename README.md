# SLM Trainer — from-scratch small language model training pipeline

Train a decoder-only transformer from scratch on mixed data (text + code + chat).
Designed to be hardware-aware and parametric — choose a model size config and
launch, the rest adapts dynamically.

## Quickstart

```bash
# 1. Set up the environment (first time only)
./setup_pod.sh

# 2. Launch the full pipeline (tokenizer + data + training)
bash scripts/launch_training.sh 300m
```

This single command will:
1. Detect hardware (GPU, CPU, RAM) and write `configs/hardware/auto.yaml`
2. Train a 32k BPE tokenizer on text + code + chat data
3. Pre-tokenize all data into binary shards
4. Launch training in a `tmux` session (survives SSH disconnects)

## Manual step-by-step

```bash
# 1. Hardware detection (generates configs/hardware/auto.yaml)
python src/hardware_detect.py

# 2. Train a tokenizer
python src/tokenizer/train_tokenizer.py --vocab-size 32768

# 3. Pre-tokenize data into binary shards
python src/data/preprocess.py

# 4. Launch training
python src/train.py --model configs/model/300m.yaml
```

## Model sizes

Three pre-computed configs are provided, all verified at ±1% of target params:

| Config | Params | n_layers | d_model | n_heads | File |
|--------|--------|----------|---------|---------|------|
| 100M   | 100.0M | 35       | 448     | 4       | `configs/model/100m.yaml` |
| 300M   | 299.7M | 13       | 1280    | 5       | `configs/model/300m.yaml` |
| 800M   | 802.8M | 35       | 1344    | 6       | `configs/model/800m.yaml` |

To search for custom sizes: `python src/model/sizing_search.py --targets 50M,200M`

## Architecture

- **Decoder-only transformer** with RMSNorm, RoPE (rotary embeddings),
  SwiGLU MLP, weight tying (embedding/output)
- **Attention**: `torch.nn.functional.scaled_dot_product_attention` (flash
  attention backend when available, no extra deps)
- **No pre-trained weights**: all parameters initialized from scratch

## Hardware detection

`src/hardware_detect.py` auto-detects and writes `configs/hardware/auto.yaml`:
  - Precision: bf16 on Ampere+, fp16 otherwise
  - Micro batch size: scaled to VRAM
  - Gradient accumulation: computed to hit ~0.5M tokens/step effective
  - DataLoader workers, gradient checkpointing, torch.compile, TF32

Override with: `python src/hardware_detect.py --overlay my_overrides.yaml`

## Monitoring

```bash
# TensorBoard
tensorboard --logdir logs/ --bind_all

# GUI inference
python gui/app.py --checkpoint-dir checkpoints --tokenizer data/tokenizer/tokenizer.json
```

Metrics: `train/loss`, `train/perplexity`, `train/lr`, `train/grad_norm`,
`train/tokens_per_sec`, `train/step_time_ms`, `train/mfu`,
`system/gpu_mem_allocated`, `system/gpu_util`, `system/dataloader_wait_ms`,
`val/loss`, `val/perplexity`.

## Benchmarks (RTX 4090, 24 GB)

| Config | Tokens/sec | MFU | GPU Mem | Note |
|--------|-----------|-----|---------|------|
| 300M, bs=4, no compile | 38,482 | 22.0% | 9.1 GB | |
| 300M, bs=4, compile(default) | 48,756 | 27.9% | 7.3 GB | |
| 300M, bs=8, compile(default) | 56,371 | 32.3% | 10.6 GB | |
| 300M, bs=9×16, compile(default) | ~49,000 | ~28% | ~8 GB | Production config (grad acc + checkpointing) |

`torch.compile(mode="default")` is used in production (+27% throughput vs no compile).
`mode="reduce-overhead"` (CUDA graphs) is incompatible with weight tying + gradient accumulation
and falls back automatically.

See `BENCHMARKS.md` for full optimization history.

## Project structure

```
slm-trainer/
  configs/
    model/   — 100m.yaml, 300m.yaml, 800m.yaml (model architecture configs)
    data/    — mixture.yaml (data mixing ratios)
    hardware/ — auto.yaml (runtime-generated, hardware detection)
  src/
    hardware_detect.py       — detects GPU/CPU/RAM, writes auto.yaml
    tokenizer/train_tokenizer.py — BPE tokenizer training
    data/                    — download, pre-tokenize, dataset loader
    model/                   — transformer layers, model, sizing search
    train.py                 — main training entry point
    eval.py                  — evaluation (perplexity + generation)
    benchmark.py             — micro-benchmark runner
  gui/app.py                 — Gradio inference interface
  scripts/launch_training.sh — one-command full pipeline launcher
  data/          (gitignored — shards tokenisés binaires)
  checkpoints/   (gitignored)
  logs/          (tensorboard — gitignored)
```

## Hardware requirements

- NVIDIA GPU with CUDA support (tested on 1x RTX 4090, 24 GB VRAM)
- Python 3.10+, PyTorch 2.x, `tokenizers`, `datasets`, `accelerate`, `gradio`
- Disk: ~25 GB for pre-tokenized data (12B tokens × 2 bytes), plus checkpoints
- Multi-GPU supported via `torchrun` (DDP)
