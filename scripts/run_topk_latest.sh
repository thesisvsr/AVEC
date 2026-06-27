#!/bin/bash
#
# Run Top-K Accuracy Analysis on the LATEST checkpoint
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Find latest checkpoint
# Search in callbacks/LipBengal/AV/VisualCE/
CHECKPOINT_DIR="callbacks/LipBengal/AV/VisualCE"
LATEST_CKPT_PATH=$(ls -t "$CHECKPOINT_DIR"/checkpoints_epoch_*.ckpt | head -n 1)

if [ -z "$LATEST_CKPT_PATH" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

LATEST_CKPT_NAME=$(basename "$LATEST_CKPT_PATH")
echo "========================================"
echo "Top-K Analysis for Latest Checkpoint"
echo "========================================"
echo "Checkpoint: $LATEST_CKPT_NAME"
echo "Dataset: LipBengal"
echo "Output Dir: topk_analysis_results/VisualCE_epoch232"
echo ""

# Run the python analysis script
python3 scripts/topk_analysis.py \
    --config_file configs/LipBengal/AV/VisualCE.py \
    --checkpoint "$LATEST_CKPT_NAME" \
    --k_values 1 3 5 10 20 \
    --output_dir topk_analysis_results/VisualCE_epoch232 \
    --cpu  # Use CPU to avoid OOM if training is somehow still hogging GPU memory (optional, remove if GPU is free)

echo ""
echo "========================================"
echo "Analysis Complete!"
echo "========================================"

