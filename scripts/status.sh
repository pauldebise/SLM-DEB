#!/usr/bin/env bash
# Quick pipeline status: pre-tokenization progress, training state, GPU state.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "=== SLM Trainer Status ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"

# Pre-tokenization
MANIFEST="data/shards/manifest.json"
SHARD_COUNT=$(ls data/shards/*.bin 2>/dev/null | wc -l || echo 0)
SHARD_SIZE=$(du -sh data/shards/ 2>/dev/null | cut -f1 || echo "N/A")
TOKENIZER="data/tokenizer/tokenizer.json"

echo ""
echo "--- Data Pipeline ---"
if [ -f "$MANIFEST" ]; then
  TRAIN_TOKENS=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m.get('total_train_tokens',0))")
  VAL_TOKENS=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m.get('total_val_tokens',0))")
  TOTAL=$((TRAIN_TOKENS + VAL_TOKENS))
  echo "Manifest: EXISTS"
  echo "Train tokens: $(python3 -c "print(f'{$TRAIN_TOKENS/1e9:.2f}B')")"
  echo "Val tokens:   $(python3 -c "print(f'{$VAL_TOKENS/1e6:.2f}M')")"
  echo "Total tokens: $(python3 -c "print(f'{$TOTAL/1e9:.2f}B')")"
  echo "Status: COMPLETE"
else
  echo "Manifest: not yet created (pre-tokenization running)"
  echo "Shards: $SHARD_COUNT files, $SHARD_SIZE"
  if [ "$SHARD_COUNT" -gt 0 ]; then
    EST_TOKENS=$((SHARD_COUNT * 10000000))
    echo "Estimated tokens: $(python3 -c "print(f'{$EST_TOKENS/1e9:.2f}B')")"
    LAST_SHARD=$(ls -t data/shards/*.bin 2>/dev/null | head -1)
    if [ -n "$LAST_SHARD" ]; then
      LAST_TIME=$(stat -c "%Y" "$LAST_SHARD" 2>/dev/null || echo 0)
      NOW=$(date +%s)
      AGE=$((NOW - LAST_TIME))
      if [ "$AGE" -lt 300 ]; then
        echo "Last shard: $(basename "$LAST_SHARD") (${AGE}s ago) — ACTIVE"
      else
        echo "Last shard: $(basename "$LAST_SHARD") ($((AGE/60))m ago) — MAYBE STALLED"
      fi
    fi
  fi
  if [ -f "$TOKENIZER" ]; then
    echo "Tokenizer: $(du -sh "$TOKENIZER" 2>/dev/null | cut -f1)"
  else
    echo "Tokenizer: MISSING"
  fi
fi

# Training
echo ""
echo "--- Training ---"
CKPT_COUNT=$(find checkpoints/ -name "*.pt" 2>/dev/null | wc -l || echo 0)
if [ "$CKPT_COUNT" -gt 0 ]; then
  echo "Checkpoints: $CKPT_COUNT"
  LATEST_CKPT=$(ls -t checkpoints/checkpoint_step_*.pt 2>/dev/null | head -1)
  if [ -n "$LATEST_CKPT" ]; then
    STEP=$(python3 -c "import torch; c=torch.load('$LATEST_CKPT', map_location='cpu', weights_only=False); print(c.get('step','?'))")
    LOSS=$(python3 -c "import torch; c=torch.load('$LATEST_CKPT', map_location='cpu', weights_only=False); print(f\"{c.get('loss_log',[0])[-1]:.2f}\" if c.get('loss_log') else '?')")
    echo "Latest: $(basename "$LATEST_CKPT") (step=$STEP, last_loss=$LOSS)"
  fi
else
  echo "No checkpoints yet"
fi

# GPU
echo ""
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r util mem_used mem_total temp; do
    echo "GPU util: ${util}%, VRAM: ${mem_used}/${mem_total} MiB, Temp: ${temp}C"
  done
else
  echo "nvidia-smi not available"
fi

# Background processes
echo ""
echo "--- Processes ---"
ps aux | grep -E "preprocess|train\.py" | grep -v grep | awk '{printf "  %s (PID %s, CPU %s, MEM %s)\n", $11, $2, $3, $4}' || echo "  None"

# tmux sessions
if command -v tmux &>/dev/null; then
  echo ""
  echo "--- tmux ---"
  tmux ls 2>/dev/null || echo "  No tmux sessions"
fi

echo ""
echo "=== Done ==="
