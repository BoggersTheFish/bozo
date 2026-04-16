#!/bin/bash
STAGE1_PID=113134
echo "Watching stage 1 (PID $STAGE1_PID)..."
while kill -0 $STAGE1_PID 2>/dev/null; do
    sleep 30
done

sleep 10
echo "[$(date)] Stage 1 done. Launching stage 2 — FineWeb-Edu 500M tokens..."

torchrun --nproc_per_node=2 /workspace/bozo/train.py \
    --data_dir /workspace/bozo/data/fineweb-edu \
    --train_tokens 500_000_000 \
    --preset large \
    --resume /workspace/bozo/checkpoints/stage1_logic_117m/latest.pt \
    --w_consistency 0.1 --w_entropy 0.05 \
    --out_dir /workspace/bozo/checkpoints/stage2_language_117m \
    --log_csv /workspace/bozo/logs/stage2_language_117m.csv \
    >> /workspace/bozo/logs/stage2_language_117m.log 2>&1

echo "[$(date)] Stage 2 complete."
