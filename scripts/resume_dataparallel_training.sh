#!/bin/bash
#
# Dual-GPU Training with DataParallel (Simpler, Better Progress Bar)
# Resume from latest checkpoint
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv"
    exit 1
fi

# Configuration
CONFIG_PATH="configs/LipBengal/AV/VisualCE.py"
CALLBACK_DIR="callbacks/LipBengal/AV/VisualCE"
LOGS_DIR="logs/dual_gpu_training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/dataparallel_${TIMESTAMP}.log"

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
echo "  DUAL-GPU TRAINING (DataParallel - Simpler & Faster)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "Strategy: PyTorch DataParallel" | tee -a "$LOG_FILE"
echo "GPUs: 2x NVIDIA GeForce RTX 3060 (12GB each)" | tee -a "$LOG_FILE"
echo "Resuming from: $CHECKPOINT_NAME (Epoch $CURRENT_EPOCH)" | tee -a "$LOG_FILE"
echo "Target: Epoch 250" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Show GPU status
echo "📊 GPU Status:" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create temporary config with modified epochs
TEMP_CONFIG="configs/LipBengal/AV/VisualCE_parallel.py"
cp "$CONFIG_PATH" "$TEMP_CONFIG"
# Set target epochs to 250
sed -i 's/^epochs = .*/epochs = 250/' "$TEMP_CONFIG"

echo "ℹ️  Training Configuration:" | tee -a "$LOG_FILE"
echo "  - Mode: DataParallel (both GPUs work together)" | tee -a "$LOG_FILE"
echo "  - Batch size per forward: 32" | tee -a "$LOG_FILE"
echo "  - GPUs split the batch automatically" | tee -a "$LOG_FILE"
echo "  - Workers: 4" | tee -a "$LOG_FILE"
echo "  - Progress bar: Visible (single process)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "▶ Starting DataParallel Training..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run with DataParallel on both GPUs
# --parallel flag enables DataParallel
# This uses a single process with visible progress bars
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --config_file "$TEMP_CONFIG" \
    --mode training \
    --checkpoint "$CHECKPOINT_NAME" \
    -j 4 \
    --parallel \
    --saving_period_epoch 1 \
    --eval_period_epoch 1 \
    --step_log_period 10 \
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

# Show final checkpoints
echo "" | tee -a "$LOG_FILE"
echo "📊 Final Checkpoints:" | tee -a "$LOG_FILE"
ls -lht "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -5 | tee -a "$LOG_FILE"

# Show GPU utilization stats
echo "" | tee -a "$LOG_FILE"
echo "📊 Final GPU Status:" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅ Script completed!" | tee -a "$LOG_FILE"







