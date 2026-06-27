#!/bin/bash
# Run missing LipBengal T2 experiments

set -e
cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="logs/missing_t2_experiments_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "         Running Missing LipBengal T2 Experiments" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to run experiment
run_experiment() {
    local EXPERIMENT=$1
    local TARGET_EPOCHS=100
    
    local CONFIG_PATH="configs/LipBengal/AV/ablations/${EXPERIMENT}.py"
    local LOG_DIR="logs/ablations/LipBengal/$EXPERIMENT"
    local CALLBACK_DIR="callbacks/LipBengal/AV/ablations/$EXPERIMENT"
    
    mkdir -p "$LOG_DIR"
    
    # Check if already complete
    local LATEST_CKPT=$(ls -v "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        local CURRENT_EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
        if [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCHS" ]; then
            echo "  ✓ Already complete: $EXPERIMENT (Epoch $CURRENT_EPOCH)" | tee -a "$LOG_FILE"
            return 0
        fi
        echo "  Resuming $EXPERIMENT from Epoch $CURRENT_EPOCH" | tee -a "$LOG_FILE"
        RESUME_ARG="--checkpoint $LATEST_CKPT"
    else
        echo "  Starting $EXPERIMENT from scratch" | tee -a "$LOG_FILE"
        RESUME_ARG=""
    fi
    
    echo "▶ Starting: LipBengal / $EXPERIMENT on GPU 0" | tee -a "$LOG_FILE"
    echo "  Start time: $(date)" | tee -a "$LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --config_file "$CONFIG_PATH" \
        --mode training \
        -j 4 \
        $RESUME_ARG \
        2>&1 | tee -a "$LOG_DIR/training_${TIMESTAMP}.log"
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Completed: $EXPERIMENT" | tee -a "$LOG_FILE"
        echo "  End time: $(date)" | tee -a "$LOG_FILE"
    else
        echo "  ✗ Failed: $EXPERIMENT (Exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        return $EXIT_CODE
    fi
    echo "" | tee -a "$LOG_FILE"
}

# Run T2_freeze_0ep
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "Experiment 1/2: T2_freeze_0ep (No encoder freezing)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
run_experiment "T2_freeze_0ep"

# Run T2_freeze_10ep
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "Experiment 2/2: T2_freeze_10ep (10 epochs encoder freezing)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
run_experiment "T2_freeze_10ep"

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "              ✅ ALL EXPERIMENTS COMPLETE!" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"




