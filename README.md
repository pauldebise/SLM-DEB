# SLM Trainer — from-scratch small language model training pipeline

Train a decoder-only transformer from scratch on mixed data (text + code + chat).
Designed to be hardware-aware and parametric — choose a model size config and
launch, the rest adapts dynamically.

## Quickstart

```bash
# 1. Set up the environment
./setup_pod.sh

# 2. Train a tokenizer (or use existing)
python src/tokenizer/train_tokenizer.py --vocab-size 32768

# 3. Download and pre-tokenize data
python src/data/download.py
python src/data/preprocess.py

# 4. Launch training
python src/train.py --model configs/model/300m.yaml
```

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
    eval.py                  — evaluation on benchmarks
    benchmark.py             — micro-benchmark runner
  gui/app.py                 — Gradio inference interface
  scripts/                   — utility scripts
```

## Hardware requirements

- NVIDIA GPU with CUDA support (tested on 1x RTX 4090, 24 GB VRAM)
- Python 3.10+, PyTorch 2.x, `tokenizers`, `datasets`, `accelerate`
- Disk: ~25 GB for pre-tokenized data (12B tokens × 2 bytes), plus checkpoints

## Monitoring

Launch TensorBoard to watch training metrics:

```bash
tensorboard --logdir logs/ --bind_all
```

Key metrics logged: loss, perplexity, learning rate, grad norm, tokens/sec,
MFU, GPU memory, dataloader wait time.
