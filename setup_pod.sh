#!/usr/bin/env bash
set -euo pipefail

echo "=== SLM Trainer pod setup ==="

if [ ! -d "/workspace/venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv /workspace/venv
fi

source /workspace/venv/bin/activate

echo "Installing/upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing core dependencies..."
pip install --quiet \
  torch \
  tokenizers \
  datasets \
  accelerate \
  gradio \
  pyyaml \
  tensorboard \
  psutil

echo "=== Setup complete ==="
echo "Run: source /workspace/venv/bin/activate"
