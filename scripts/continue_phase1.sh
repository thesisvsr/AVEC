#!/bin/bash
#
# Continue Phase 1 Experiments
# Monitors and sequentially runs remaining experiments
#

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Function to check if an experiment is complete
is_complete() {
    local DATASET=$1
    local EXPERIMENT=$2
    local LOG_PATTERN="logs/ablations/$DATASET/$EXPERIMENT/*.log"
    
    # Check for completion marker in logs
    if ls $LOG_PATTERN 1>/dev/null 2>&1; then
        local LATEST_LOG=$(ls -t $LOG_PATTERN 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            # Look for "Epoch 100/100" completion
            if grep -q "Epoch 100/100" "$LATEST_LOG" 2>/dev/null; then
                return 0  # Complete
            fi
        fi
    fi
    return 1  # Not complete
}

# Function to check if an experiment is running
is_running() {
    local DATASET=$1
    local EXPERIMENT=$2
    local CONFIG_PATTERN="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"
    
    if ps aux | grep "main.py" | grep -q "$CONFIG_PATTERN"; then
        return 0  # Running
    fi
    return 1  # Not running
}

# Function to run an experiment
run_experiment() {
    local DATASET=$1
    local EXPERIMENT=$2
    local GPU=$3
    local TARGET_EPOCHS=100
    
    echo ""
    echo "==================================================================="
    echo "▶ STARTING: $DATASET / $EXPERIMENT on GPU $GPU"
    echo "==================================================================="
    
    local CONFIG_PATH="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"
    local LOG_DIR="logs/ablations/$DATASET/$EXPERIMENT"
    local TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    local LOG_FILE="$LOG_DIR/training_auto_${TIMESTAMP}.log"
    local CALLBACK_PATH="callbacks/$DATASET/AV/ablations/$EXPERIMENT"
    
    mkdir -p "$LOG_DIR"
    
    # Check for resume
    local LATEST_CHECKPOINT=$(ls -v "$CALLBACK_PATH"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    local RESUME_ARG=""
    if [ -f "$LATEST_CHECKPOINT" ]; then
        RESUME_ARG="--resume_checkpoint $LATEST_CHECKPOINT"
        echo "📂 Resuming from: $(basename "$LATEST_CHECKPOINT")"
    else
        echo "   Starting from scratch"
    fi
    
    # Backup and modify config
    cp "$CONFIG_PATH" "${CONFIG_PATH}.bak"
    sed -i "s/^epochs = .*/epochs = $TARGET_EPOCHS/" "$CONFIG_PATH"
    
    # Run in background
    nohup bash -c "
    cd $PROJECT_ROOT
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
        --config_file $CONFIG_PATH \
        --mode training \
        -j 4 \
        $RESUME_ARG \
        2>&1 | tee -a $LOG_FILE
    mv ${CONFIG_PATH}.bak $CONFIG_PATH
    " > /dev/null 2>&1 &
    
    local PID=$!
    echo "  PID: $PID"
    echo "  Log: $LOG_FILE"
    echo "==================================================================="
    
    # Wait a moment and verify startup
    sleep 5
    if ps -p $PID > /dev/null; then
        echo "✓ Started successfully"
    else
        echo "✗ Failed to start - check log: $LOG_FILE"
    fi
}

# Function to wait for experiment completion
wait_for_completion() {
    local DATASET=$1
    local EXPERIMENT=$2
    local CHECK_INTERVAL=60  # Check every minute
    
    echo ""
    echo "⏳ Waiting for $DATASET / $EXPERIMENT to complete..."
    
    while true; do
        if is_complete "$DATASET" "$EXPERIMENT"; then
            echo "✓ $DATASET / $EXPERIMENT completed!"
            return 0
        fi
        
        if ! is_running "$DATASET" "$EXPERIMENT"; then
            echo "⚠️  $DATASET / $EXPERIMENT not running (may have crashed)"
            return 1
        fi
        
        sleep $CHECK_INTERVAL
    done
}

echo "==================================================================="
echo "         Phase 1 Continuation Script"
echo "==================================================================="
echo ""
echo "Checking current status..."

# Check LipBengal experiments
echo ""
echo "--- LipBengal Pipeline (GPU 0) ---"

if is_complete "LipBengal" "S1_raw"; then
    echo "✓ S1_raw: Complete"
else
    echo "⏳ S1_raw: Running (will wait for completion)"
    wait_for_completion "LipBengal" "S1_raw"
fi

if is_complete "LipBengal" "S1_simple"; then
    echo "✓ S1_simple: Complete"
elif is_running "LipBengal" "S1_simple"; then
    echo "⏳ S1_simple: Running (will wait for completion)"
    wait_for_completion "LipBengal" "S1_simple"
else
    run_experiment "LipBengal" "S1_simple" 0
    wait_for_completion "LipBengal" "S1_simple"
fi

if is_complete "LipBengal" "S1_mixed"; then
    echo "✓ S1_mixed: Complete"
elif is_running "LipBengal" "S1_mixed"; then
    echo "⏳ S1_mixed: Running"
else
    run_experiment "LipBengal" "S1_mixed" 0
fi

# Check LRW-AR experiments
echo ""
echo "--- LRW-AR Pipeline (GPU 1) ---"

# S1_raw was stopped by user, skip it
echo "⊗ S1_raw: Stopped (as requested)"

if is_complete "LRW-AR" "S1_simple"; then
    echo "✓ S1_simple: Complete"
elif is_running "LRW-AR" "S1_simple"; then
    echo "⏳ S1_simple: Running (will wait for completion)"
    wait_for_completion "LRW-AR" "S1_simple"
else
    run_experiment "LRW-AR" "S1_simple" 1
    wait_for_completion "LRW-AR" "S1_simple"
fi

if is_complete "LRW-AR" "S1_mixed"; then
    echo "✓ S1_mixed: Complete"
elif is_running "LRW-AR" "S1_mixed"; then
    echo "⏳ S1_mixed: Running"
else
    run_experiment "LRW-AR" "S1_mixed" 1
fi

echo ""
echo "==================================================================="
echo "         Phase 1 Continuation Complete!"
echo "==================================================================="
echo ""
echo "Final Status:"
echo "  GPU 0 (LipBengal): S1_raw → S1_simple → S1_mixed"
echo "  GPU 1 (LRW-AR):    S1_simple → S1_mixed (S1_raw stopped)"
echo ""
echo "All scheduled Phase 1 experiments are now running or completed!"
echo "==================================================================="

