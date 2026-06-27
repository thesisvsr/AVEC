#!/bin/bash
#
# Run S1_raw experiments in PARALLEL for both LipBengal and LRW-AR
#

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# GPU assignment
LIPBENGAL_GPU=${1:-0}
LRWAR_GPU=${2:-1}

echo "==================================================================="
echo "Running S1_raw experiments in PARALLEL"
echo "==================================================================="
echo "Dataset 1: LipBengal → GPU $LIPBENGAL_GPU"
echo "Dataset 2: LRW-AR    → GPU $LRWAR_GPU"
echo "==================================================================="
echo ""

# Function to run training
run_experiment() {
    local DATASET=$1
    local GPU=$2
    local TARGET_EPOCHS=$3
    local EXPERIMENT="S1_raw"
    
    echo "▶ STARTING: $DATASET/$EXPERIMENT on GPU $GPU"
    
    CONFIG_PATH="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "✗ Config not found: $CONFIG_PATH"
        return 1
    fi
    
    # Set epochs in config
    cp "$CONFIG_PATH" "${CONFIG_PATH}.bak.$$"
    sed -i "s/^epochs = .*/epochs = $TARGET_EPOCHS/" "$CONFIG_PATH"
    
    # Launch training
    LOG_DIR="logs/ablations/${DATASET}/${EXPERIMENT}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/training_parallel_$(date +%Y%m%d_%H%M%S).log"
    
    CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
        --config_file "$CONFIG_PATH" \
        --mode training \
        -j 4 \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    # Restore original config
    mv "${CONFIG_PATH}.bak.$$" "$CONFIG_PATH"
    
    echo ""
    echo "==================================================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ COMPLETED: $DATASET/$EXPERIMENT"
    else
        echo "✗ FAILED: $DATASET/$EXPERIMENT (exit code: $EXIT_CODE)"
    fi
    echo "==================================================================="
    
    return $EXIT_CODE
}

# Export function for use in subshells
export -f run_experiment
export PROJECT_ROOT

# Launch both experiments in parallel
echo "Launching LipBengal S1_raw on GPU $LIPBENGAL_GPU..."
run_experiment "LipBengal" "$LIPBENGAL_GPU" 100 &
LIPBENGAL_PID=$!

echo "Launching LRW-AR S1_raw on GPU $LRWAR_GPU..."
run_experiment "LRW-AR" "$LRWAR_GPU" 100 &
LRWAR_PID=$!

echo ""
echo "==================================================================="
echo "Both experiments launched!"
echo "LipBengal PID: $LIPBENGAL_PID (GPU $LIPBENGAL_GPU)"
echo "LRW-AR    PID: $LRWAR_PID    (GPU $LRWAR_GPU)"
echo "==================================================================="
echo ""
echo "Monitoring progress..."
echo "- LipBengal logs: logs/ablations/LipBengal/S1_raw/"
echo "- LRW-AR logs:    logs/ablations/LRW-AR/S1_raw/"
echo ""

# Wait for both to complete
wait $LIPBENGAL_PID
LIPBENGAL_EXIT=$?

wait $LRWAR_PID
LRWAR_EXIT=$?

echo ""
echo "==================================================================="
echo "PARALLEL EXECUTION COMPLETED"
echo "==================================================================="
echo "LipBengal S1_raw: $([ $LIPBENGAL_EXIT -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "LRW-AR S1_raw:    $([ $LRWAR_EXIT -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "==================================================================="

# Exit with error if either failed
if [ $LIPBENGAL_EXIT -ne 0 ] || [ $LRWAR_EXIT -ne 0 ]; then
    exit 1
fi

exit 0


