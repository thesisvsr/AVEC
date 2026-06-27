#!/bin/bash
#
# Smart Phase 1 Runner - 100 epochs, skip completed
#

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

DATASET="LipBengal"
TARGET_EPOCHS=100
GPU=${1:-0}

echo "==================================================================="
echo "Phase 1: Script Normalization Experiments"
echo "Target: $TARGET_EPOCHS epochs per experiment"
echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "==================================================================="
echo ""

# List of Phase 1 experiments
EXPERIMENTS=("S1_raw" "S1_phonetic" "S1_simple" "S1_mixed")

for EXP_ID in "${EXPERIMENTS[@]}"; do
    echo "-------------------------------------------------------------------"
    echo "Checking: $EXP_ID"
    
    # Check if already completed
    CKPT_DIR="callbacks/$DATASET/AV/ablations/$EXP_ID"
    if [ -d "$CKPT_DIR" ]; then
        LATEST_CKPT=$(ls -t "$CKPT_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            CURRENT_EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
            if [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCHS" ]; then
                echo "✓ SKIPPED - Already completed ($CURRENT_EPOCH epochs)"
                echo ""
                continue
            fi
        fi
    fi
    
    # Create/update config with 100 epochs
    CONFIG_PATH="configs/$DATASET/AV/ablations/${EXP_ID}.py"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "✗ Config not found: $CONFIG_PATH"
        continue
    fi
    
    # Backup original and set to 100 epochs
    cp "$CONFIG_PATH" "${CONFIG_PATH}.bak"
    sed -i "s/^epochs = .*/epochs = $TARGET_EPOCHS/" "$CONFIG_PATH"
    
    echo "▶ STARTING: $EXP_ID (target: $TARGET_EPOCHS epochs)"
    
    # Launch training
    LOG_FILE="logs/ablations/${DATASET}/${EXP_ID}/training.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
        --config_file "$CONFIG_PATH" \
        --mode training \
        -j 4 \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    # Restore original config
    mv "${CONFIG_PATH}.bak" "$CONFIG_PATH"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ COMPLETED: $EXP_ID"
    else
        echo "✗ FAILED: $EXP_ID (exit code: $EXIT_CODE)"
    fi
    
    echo ""
    
    # Clean old checkpoints to save space
    if [ -d "$CKPT_DIR" ]; then
        ls "$CKPT_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null \
            | grep -v -E "epoch_(10|20|30|40|50|60|70|80|90|100)_" \
            | xargs -r rm 2>/dev/null || true
    fi
done

echo "==================================================================="
echo "Phase 1 Complete!"
echo "==================================================================="


