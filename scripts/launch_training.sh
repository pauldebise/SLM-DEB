#!/usr/bin/env bash
# Full pipeline launch script for the SLM trainer.
# Usage: bash scripts/launch_training.sh [100m|300m|800m] [--total-tokens N]
#
# This script:
# 1. Detects hardware and generates auto.yaml
# 2. Trains the tokenizer (if not exists)
# 3. Downloads and pre-tokenizes data (if not exists)
# 4. Launches training in a tmux session (so it survives SSH disconnects)
#
# Examples:
#   bash scripts/launch_training.sh 300m                    # default ~12B tokens
#   bash scripts/launch_training.sh 300m --total-tokens 1B  # ~1B tokens (quick)
#   bash scripts/launch_training.sh 300m --total-tokens 12B # ~12B tokens

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

MODEL_SIZE="${1:-300m}"
TOTAL_TOKENS="12B"
if [ $# -ge 3 ] && [ "$2" = "--total-tokens" ]; then
  TOTAL_TOKENS="$3"
fi

MODEL_CONFIG="configs/model/${MODEL_SIZE}.yaml"
TOKENIZER_PATH="data/tokenizer/tokenizer.json"
MANIFEST_PATH="data/shards/manifest.json"
SESSION_NAME="slm-train-${MODEL_SIZE}"

TOTAL_TOKENS_NUM=$(python3 -c "
t = '${TOTAL_TOKENS}'.upper()
mult = 1
if t.endswith('B'):
    mult, t = 1_000_000_000, t[:-1]
elif t.endswith('M'):
    mult, t = 1_000_000, t[:-1]
elif t.endswith('K'):
    mult, t = 1_000, t[:-1]
val = float(t) * mult
print(int(val))
")

echo "=== SLM Trainer Pipeline ==="
echo "Model: ${MODEL_SIZE}"
echo "Total tokens target: ${TOTAL_TOKENS} (~${TOTAL_TOKENS_NUM} tokens)"
echo "Repo: ${REPO_DIR}"
echo ""

# Step 1: Hardware detection
echo "[1/4] Detecting hardware..."
python3 src/hardware_detect.py --output configs/hardware/auto.yaml

# Step 2: Tokenizer
if [ ! -f "${TOKENIZER_PATH}" ]; then
  echo "[2/4] Training tokenizer (32k vocab)..."
  python3 src/tokenizer/train_tokenizer.py \
    --vocab-size 32768 \
    --output "${TOKENIZER_PATH}" \
    --text-samples 200000 \
    --code-samples 50000 \
    --chat-samples 50000
else
  echo "[2/4] Tokenizer already exists at ${TOKENIZER_PATH}"
fi

# Step 3: Pre-tokenize data
if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "[3/4] Pre-tokenizing data..."
  python3 src/data/preprocess.py \
    --config configs/data/mixture.yaml \
    --tokenizer "${TOKENIZER_PATH}" \
    --output-dir data/shards \
    --max-tokens "${TOTAL_TOKENS_NUM}"
else
  echo "[3/4] Pre-tokenized data already exists at ${MANIFEST_PATH}"
fi

# Step 4: Launch training
echo "[4/4] Launching training for ${MODEL_SIZE}..."

# Estimate total steps
EFFECTIVE_TOKENS=$(python3 -c "
import yaml
with open('configs/hardware/auto.yaml') as f:
    hw = yaml.safe_load(f)
target = hw['batch']['target_effective_tokens_per_step']
print(target)
")
MAX_STEPS=$((TOTAL_TOKENS_NUM / EFFECTIVE_TOKENS))
echo "Estimated max steps for ~${TOTAL_TOKENS} tokens: ${MAX_STEPS}"

# Launch in tmux if available
TRAIN_CMD="python3 src/train.py \
  --model ${MODEL_CONFIG} \
  --hardware configs/hardware/auto.yaml \
  --manifest ${MANIFEST_PATH} \
  --checkpoint-dir checkpoints \
  --log-dir logs \
  --max-steps ${MAX_STEPS} \
  --lr 3e-4 \
  --warmup-steps 2000 \
  --log-interval 10 \
  --val-interval 1000 \
  --save-interval 5000 \
  --keep-checkpoints 3"

if command -v tmux &> /dev/null; then
  echo "Starting in tmux session: ${SESSION_NAME}"
  tmux new-session -d -s "${SESSION_NAME}" "${TRAIN_CMD}"
  echo "Training started in background tmux session."
  echo "  Attach: tmux attach -t ${SESSION_NAME}"
  echo "  Detach: Ctrl+B then D"
  echo "  List:   tmux ls"
  echo ""
  echo "Monitor with: tensorboard --logdir logs/ --bind_all"
  echo "Inference GUI: python3 gui/app.py --checkpoint-dir checkpoints --tokenizer ${TOKENIZER_PATH}"
else
  echo "tmux not available, running in foreground:"
  echo "${TRAIN_CMD}"
  echo ""
  exec ${TRAIN_CMD}
fi
