#!/bin/bash
#
# Run ONLY S1_raw experiment (100 epochs)
#

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

DATASET="LipBengal"
EXPERIMENT="S1_raw"
TARGET_EPOCHS=100
GPU=${1:-0}

echo "==================================================================="
echo "Running S1_raw - Raw Bengali Script (Baseline)"
echo "==================================================================="
echo "Experiment: $EXPERIMENT"
echo "Dataset: $DATASET"
echo "Target Epochs: $TARGET_EPOCHS"
echo "GPU: $GPU"
echo "Label Format: raw (Bengali script: অধ্যয়ন, অনুভব, অন্ধকার...)"
echo "Expected Result: ~1-2% accuracy (to prove script normalization is essential)"
echo "==================================================================="
echo ""

# Config path
CONFIG_PATH="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "✗ Config not found: $CONFIG_PATH"
    exit 1
fi

# Set epochs to 100 in config
cp "$CONFIG_PATH" "${CONFIG_PATH}.bak"
sed -i "s/^epochs = .*/epochs = $TARGET_EPOCHS/" "$CONFIG_PATH"

echo "▶ STARTING: $EXPERIMENT"
echo ""

# Launch training
LOG_DIR="logs/ablations/${DATASET}/${EXPERIMENT}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training.log"

CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
    --config_file "$CONFIG_PATH" \
    --mode training \
    -j 4 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Restore original config
mv "${CONFIG_PATH}.bak" "$CONFIG_PATH"

echo ""
echo "==================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ COMPLETED: $EXPERIMENT"
else
    echo "✗ FAILED: $EXPERIMENT (exit code: $EXIT_CODE)"
fi
echo "==================================================================="


