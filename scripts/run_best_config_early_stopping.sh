#!/bin/bash
# Run Best LipBengal Configuration with Early Stopping

set -e
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Activated virtual environment"
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="logs/best_config_early_stopping_${TIMESTAMP}.log"
CONFIG_PATH="configs/LipBengal/AV/best_config_early_stopping.py"
CALLBACK_DIR="callbacks/LipBengal/AV/best_config_early_stopping"

mkdir -p logs
mkdir -p "$CALLBACK_DIR"

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  🚀 BEST LIPBENGAL CONFIGURATION - TRAINING WITH EARLY STOPPING" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Configuration: T1_frontend (Best: 37.07% baseline accuracy)" | tee -a "$LOG_FILE"
echo "Strategy: Frontend Transfer Learning" | tee -a "$LOG_FILE"
echo "Early Stopping: Enabled (patience=15, min_delta=0.001)" | tee -a "$LOG_FILE"
echo "GPU: CUDA:0" | tee -a "$LOG_FILE"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if already training
LATEST_CKPT=$(ls -v "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    CURRENT_EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
    echo "▶ Found existing checkpoint: Epoch $CURRENT_EPOCH" | tee -a "$LOG_FILE"
    echo "▶ Resuming training from checkpoint..." | tee -a "$LOG_FILE"
    RESUME_ARG="--checkpoint $LATEST_CKPT"
else
    echo "▶ Starting training from scratch..." | tee -a "$LOG_FILE"
    RESUME_ARG=""
fi

echo "" | tee -a "$LOG_FILE"

# Start training
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --config_file "$CONFIG_PATH" \
    --mode training \
    -j 4 \
    $RESUME_ARG \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ TRAINING COMPLETED SUCCESSFULLY" | tee -a "$LOG_FILE"
else
    echo "  ⚠️  TRAINING STOPPED (Exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
fi
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "End Time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Checkpoints saved to: $CALLBACK_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Show latest checkpoint info
if [ -f "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt ]; then
    echo "📊 Latest Checkpoint:" | tee -a "$LOG_FILE"
    ls -lht "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt | head -3 | tee -a "$LOG_FILE"
fi

