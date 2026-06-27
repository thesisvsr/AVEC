#!/bin/bash
#
# Run Comprehensive Top-K Analysis on LipBengal Checkpoint Epoch 232
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

# Configuration
CHECKPOINT="callbacks/LipBengal/AV/VisualCE/checkpoints_epoch_232_step_317941.ckpt"
CONFIG="configs/LipBengal/AV/VisualCE.py"
OUTPUT_DIR="topk_analysis_results/LipBengal_Epoch232"

echo "================================================================="
echo "Starting Top-K Analysis for Epoch 232"
echo "================================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output Dir: $OUTPUT_DIR"
echo "================================================================="

# Run analysis
python3 scripts/run_topk_and_metrics.py \
    --config_file "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --k_values 1 2 3 4 5 10

echo ""
echo "================================================================="
echo "✅ Analysis Completed!"
echo "Check results in: $OUTPUT_DIR"
echo "  - results.json"
echo "  - topk_plot.png"
echo "================================================================="

