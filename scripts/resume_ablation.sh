#!/bin/bash
#
# Resume Ablation Training with Custom Epochs
#
# Usage:
#   ./scripts/resume_ablation.sh S1_phonetic LipBengal 150 0
#

set -e

EXP_ID=$1
DATASET=$2
TOTAL_EPOCHS=${3:-150}
GPU=${4:-0}

if [ -z "$EXP_ID" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 <exp_id> <dataset> [total_epochs] [gpu]"
    echo "Example: $0 S1_phonetic LipBengal 150 0"
    exit 1
fi

# Find latest checkpoint
CKPT_DIR="callbacks/$DATASET/AV/ablations/$EXP_ID"
LATEST_CKPT=$(ls -t "$CKPT_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Error: No checkpoint found in $CKPT_DIR"
    exit 1
fi

# Extract current epoch
CURRENT_EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
echo "Found checkpoint at epoch $CURRENT_EPOCH"
echo "Will train until epoch $TOTAL_EPOCHS"

if [ "$CURRENT_EPOCH" -ge "$TOTAL_EPOCHS" ]; then
    echo "Already completed $CURRENT_EPOCH epochs (target: $TOTAL_EPOCHS)"
    exit 0
fi

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Get config path
CONFIG_PATH="configs/$DATASET/AV/ablations/${EXP_ID}.py"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

# Create temporary config with modified epochs
TEMP_CONFIG="configs/$DATASET/AV/ablations/${EXP_ID}_resume.py"
cp "$CONFIG_PATH" "$TEMP_CONFIG"
sed -i "s/^epochs = .*/epochs = $TOTAL_EPOCHS/" "$TEMP_CONFIG"

echo ""
echo "=== Resuming Training ==="
echo "Experiment: $EXP_ID"
echo "Dataset: $DATASET"
echo "Config: $TEMP_CONFIG"
echo "Checkpoint: $LATEST_CKPT"
echo "Current epoch: $CURRENT_EPOCH"
echo "Target epoch: $TOTAL_EPOCHS"
echo "GPU: $GPU"
echo ""

# Resume training
CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
    --config_file "$TEMP_CONFIG" \
    --mode training \
    --checkpoint "$LATEST_CKPT" \
    -j 4

# Cleanup
rm -f "$TEMP_CONFIG"


