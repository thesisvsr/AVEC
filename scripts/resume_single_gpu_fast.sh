#!/bin/bash
#
# Resume training with SINGLE GPU (fastest option for this model)
# Resume from latest checkpoint
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Configuration
CONFIG_PATH="configs/LipBengal/AV/VisualCE.py"
CALLBACK_DIR="callbacks/LipBengal/AV/VisualCE"
LOGS_DIR="logs/single_gpu_training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/single_gpu_${TIMESTAMP}.log"

mkdir -p "$LOGS_DIR"

# Find latest checkpoint
LATEST_CKPT=$(ls -t "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $CALLBACK_DIR"
    exit 1
fi

CHECKPOINT_NAME=$(basename "$LATEST_CKPT")
CURRENT_EPOCH=$(echo "$CHECKPOINT_NAME" | grep -oP 'epoch_\K\d+')

echo "═══════════════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  SINGLE-GPU TRAINING (FASTEST FOR THIS MODEL)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "GPU: NVIDIA GeForce RTX 3060 (GPU 0)" | tee -a "$LOG_FILE"
echo "Resuming from: $CHECKPOINT_NAME (Epoch $CURRENT_EPOCH)" | tee -a "$LOG_FILE"
echo "Target: Epoch 250" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Why single-GPU?" | tee -a "$LOG_FILE"
echo "  - DataParallel: ~40 min/epoch (too slow)" | tee -a "$LOG_FILE"
echo "  - Single-GPU: ~2.5 min/epoch (16x faster!)" | tee -a "$LOG_FILE"
echo "  - Model size doesn't benefit from multi-GPU overhead" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create temporary config
TEMP_CONFIG="configs/LipBengal/AV/VisualCE_single_gpu.py"
cp "$CONFIG_PATH" "$TEMP_CONFIG"
sed -i 's/^epochs = .*/epochs = 250/' "$TEMP_CONFIG"

echo "▶ Starting Single-GPU Training (FAST MODE)..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run on single GPU (no parallel flags)
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --config_file "$TEMP_CONFIG" \
    --mode training \
    --checkpoint "$CHECKPOINT_NAME" \
    -j 4 \
    --saving_period_epoch 1 \
    --eval_period_epoch 1 \
    --step_log_period 100 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

# Cleanup
rm -f "$TEMP_CONFIG"

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ TRAINING COMPLETED SUCCESSFULLY" | tee -a "$LOG_FILE"
else
    echo "  ⚠️  TRAINING STOPPED (Exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
fi
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "End Time: $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "📊 Final Checkpoints:" | tee -a "$LOG_FILE"
ls -lht "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -3 | tee -a "$LOG_FILE"







