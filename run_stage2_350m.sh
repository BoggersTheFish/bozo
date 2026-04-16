#!/bin/bash
# Auto-launches 350M stage 2 (open-web-math, vocab=32768) after stage 1 finishes.
# Stage 1 PID is passed as $1, or detected from the log if omitted.
#
# Vocab expansion:  stage 1 trains with vocab=16384 (logic-stage1).
#                   Stage 2 uses vocab=32768 (open-web-math).
#                   train.py detects the mismatch and auto-expands the embedding.
#
# Usage:
#   bash run_stage2_350m.sh <stage1_pid>   # watch a known PID
#   bash run_stage2_350m.sh                # poll for stage1 checkpoint

STAGE1_PID="$1"
STAGE1_DIR="/workspace/bozo/checkpoints/350m-stage1-bio"
STAGE2_DIR="/workspace/bozo/checkpoints/350m-stage2-bio"
LOG="/workspace/bozo/logs/350m_stage2_bio.log"
CSV="/workspace/bozo/logs/350m_stage2_bio.csv"

mkdir -p "$STAGE2_DIR" "$(dirname $LOG)"

if [ -n "$STAGE1_PID" ]; then
    echo "[$(date)] Watching stage 1 (PID $STAGE1_PID)..."
    while kill -0 "$STAGE1_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[$(date)] PID $STAGE1_PID exited."
else
    echo "[$(date)] Polling for stage 1 checkpoint..."
    while [ ! -f "$STAGE1_DIR/latest.pt" ]; do
        sleep 60
    done
    # Wait an extra 30 s for the file to finish writing
    sleep 30
    echo "[$(date)] Stage 1 checkpoint detected."
fi

# Copy the stage 1 checkpoint into the stage 2 dir so --resume picks it up
cp "$STAGE1_DIR/latest.pt" "$STAGE2_DIR/latest.pt"
echo "[$(date)] Checkpoint copied → $STAGE2_DIR/latest.pt"

echo "[$(date)] Launching stage 2 — open-web-math 1.1B tokens, vocab=32768..."

torchrun --nproc_per_node=2 /workspace/bozo/train.py \
    --preset 350m \
    --vocab_size 32768 \
    --data_dir /workspace/bozo/data/open-web-math \
    --train_tokens 1_100_000_000 \
    --resume \
    --out_dir "$STAGE2_DIR" \
    --decouple_optim --qk_lr_mult 3.0 \
    --sparse_grad --sparse_threshold 0.15 \
    --sleep_every 500 --sleep_steps 20 --sleep_lr_mult 0.3 \
    --ff_mode --ff_weight 0.1 \
    --w_consistency 0.05 --w_entropy 0.02 \
    --logic_mix 0.10 \
    --logic_dir /workspace/bozo/data/logic-stage1 \
    --log_every 100 --eval_every 1000 --save_every 2000 \
    --log_csv "$CSV" \
    >> "$LOG" 2>&1

echo "[$(date)] Stage 2 complete."
