#!/usr/bin/env bash
# Waits for the preprocess to finish, then syncs the manifest and restarts
# training with the full dataset.
#
# Usage: nohup bash scripts/restart_with_full_data.sh <preprocess_pid> &
#        or:  bash scripts/restart_with_full_data.sh 40414

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

PREPROCESS_PID="${1:-}"
SESSION="slm-train-300m"
LOG_FILE="${REPO_DIR}/logs/restart_watcher.log"

if [ -z "${PREPROCESS_PID}" ]; then
    echo "Usage: $0 <preprocess_pid>"
    echo "  Watches the given preprocess PID, waits for it to exit,"
    echo "  then syncs the manifest and restarts training."
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

log_msg() {
    echo "[restart_watcher $$] $(date -u): $1" | tee -a "$LOG_FILE"
}

log_msg "Starting restart watcher. Watching preprocess PID ${PREPROCESS_PID}..."

# Wait for preprocess to finish
while kill -0 "${PREPROCESS_PID}" 2>/dev/null; do
    sleep 30
done

log_msg "Preprocess (PID ${PREPROCESS_PID}) has exited. Waiting 15s for files to settle..."
sleep 15

# Sync the manifest from all shards on disk
log_msg "Syncing manifest from current shards..."
python3 scripts/sync_manifest.py \
    --tokenizer data/tokenizer/tokenizer.json \
    --shard-dir data/shards \
    --output data/shards/manifest.json 2>&1 | tee -a "$LOG_FILE"

# Show what we got
python3 -c "
import json
m = json.load(open('$REPO_DIR/data/shards/manifest.json'))
print(f'  Total: {m[\"total_train_tokens\"]/1e9:.2f}B train, {m[\"total_val_tokens\"]/1e6:.0f}M val', flush=True)
sources = {}
for s in m['train_shards']:
    src = s['source']
    sources[src] = sources.get(src, 0) + s['num_tokens']
for src, tok in sorted(sources.items()):
    print(f'  [{src}] {tok/1e9:.2f}B', flush=True)
"

# Kill old training session
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    log_msg "Stopping old training session '${SESSION}'..."
    tmux send-keys -t "${SESSION}" C-c || true
    sleep 5
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    log_msg "Old session stopped."
else
    log_msg "No existing training session found."
fi

# Start new training
log_msg "Launching new training with full dataset..."
bash scripts/launch_training.sh 300m --total-tokens 12B 2>&1 | tee -a "$LOG_FILE"

log_msg "Restart complete. Use 'tmux attach -t ${SESSION}' to monitor."
