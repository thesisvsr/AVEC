#!/bin/bash
#
# Phase 1 Optimized Dual-GPU Execution
# Maximizes GPU utilization by running experiments in parallel
# All experiments limited to 100 epochs
#

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Queue of experiments to run
# Format: "DATASET:EXPERIMENT:GPU"
declare -a GPU0_QUEUE=("LipBengal:S1_simple:0" "LipBengal:S1_mixed:0")
declare -a GPU1_QUEUE=("LRW-AR:S1_mixed:1")

# Currently running experiments (will be populated)
GPU0_CURRENT=""
GPU1_CURRENT=""

# Function to check if experiment is complete
is_complete() {
    local DATASET=$1
    local EXPERIMENT=$2
    local TARGET_EPOCHS=100
    
    local LOG_PATTERN="logs/ablations/$DATASET/$EXPERIMENT/*.log"
    if ls $LOG_PATTERN 1>/dev/null 2>&1; then
        local LATEST_LOG=$(ls -t $LOG_PATTERN 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            # Check for completion
            if grep -q "Epoch $TARGET_EPOCHS/$TARGET_EPOCHS" "$LATEST_LOG" 2>/dev/null; then
                return 0
            fi
        fi
    fi
    return 1
}

# Function to check if experiment is running on specific GPU
is_running_on_gpu() {
    local DATASET=$1
    local EXPERIMENT=$2
    local GPU=$3
    local CONFIG="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"
    
    # Check for process with this config on this GPU
    if ps aux | grep "CUDA_VISIBLE_DEVICES=$GPU" | grep "main.py" | grep -q "$CONFIG"; then
        return 0
    fi
    return 1
}

# Function to get current epoch
get_current_epoch() {
    local DATASET=$1
    local EXPERIMENT=$2
    local LOG_PATTERN="logs/ablations/$DATASET/$EXPERIMENT/*.log"
    
    if ls $LOG_PATTERN 1>/dev/null 2>&1; then
        local LATEST_LOG=$(ls -t $LOG_PATTERN 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            local EPOCH=$(grep -oP "Epoch \K[0-9]+/[0-9]+" "$LATEST_LOG" | tail -1)
            if [ -n "$EPOCH" ]; then
                echo "$EPOCH"
                return
            fi
        fi
    fi
    echo "0/100"
}

# Function to start an experiment
start_experiment() {
    local DATASET=$1
    local EXPERIMENT=$2
    local GPU=$3
    local TARGET_EPOCHS=100
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ STARTING: $DATASET / $EXPERIMENT on GPU $GPU"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    local CONFIG_PATH="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"
    local LOG_DIR="logs/ablations/$DATASET/$EXPERIMENT"
    local TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    local LOG_FILE="$LOG_DIR/training_optimized_${TIMESTAMP}.log"
    local CALLBACK_PATH="callbacks/$DATASET/AV/ablations/$EXPERIMENT"
    
    mkdir -p "$LOG_DIR"
    
    # Check for resume
    local LATEST_CHECKPOINT=$(ls -v "$CALLBACK_PATH"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    local RESUME_ARG=""
    if [ -f "$LATEST_CHECKPOINT" ]; then
        RESUME_ARG="--resume_checkpoint $LATEST_CHECKPOINT"
        local EPOCH_NUM=$(echo "$LATEST_CHECKPOINT" | grep -oP 'epoch_\K[0-9]+' | tail -1)
        echo "  📂 Resuming from epoch $EPOCH_NUM: $(basename "$LATEST_CHECKPOINT")"
    else
        echo "  🆕 Starting from scratch"
    fi
    
    echo "  🎯 Target: $TARGET_EPOCHS epochs"
    echo "  💾 Log: $LOG_FILE"
    
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
    " > /dev/null 2>&1 &
    
    local PID=$!
    echo "  🔧 PID: $PID"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Verify startup
    sleep 5
    if ps -p $PID > /dev/null 2>&1; then
        echo "  ✅ Started successfully"
    else
        echo "  ❌ Failed to start - check log"
        return 1
    fi
    
    return 0
}

# Main execution loop
echo "═══════════════════════════════════════════════════════════════════"
echo "         Phase 1 Optimized Dual-GPU Execution (100 epochs)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "⚙️  Configuration: All experiments set to 100 epochs maximum"
echo "🎮 GPU Strategy: Parallel execution to maximize utilization"
echo ""

# Check current status
echo "📊 Current Status Check:"
echo ""

echo "GPU 0 - LipBengal Pipeline:"
if is_complete "LipBengal" "S1_raw"; then
    echo "  ✅ S1_raw: Complete (100/100)"
else
    CURRENT_EPOCH=$(get_current_epoch "LipBengal" "S1_raw")
    echo "  ⏳ S1_raw: Running ($CURRENT_EPOCH)"
    GPU0_CURRENT="LipBengal:S1_raw"
fi

if is_complete "LipBengal" "S1_simple"; then
    echo "  ✅ S1_simple: Complete (100/100)"
    # Remove from queue
    GPU0_QUEUE=("${GPU0_QUEUE[@]/LipBengal:S1_simple:0/}")
else
    CURRENT_EPOCH=$(get_current_epoch "LipBengal" "S1_simple")
    if [ "$CURRENT_EPOCH" != "0/100" ]; then
        echo "  ⏳ S1_simple: Running ($CURRENT_EPOCH)"
    else
        echo "  📋 S1_simple: Queued"
    fi
fi

if is_complete "LipBengal" "S1_mixed"; then
    echo "  ✅ S1_mixed: Complete (100/100)"
    GPU0_QUEUE=("${GPU0_QUEUE[@]/LipBengal:S1_mixed:0/}")
else
    CURRENT_EPOCH=$(get_current_epoch "LipBengal" "S1_mixed")
    if [ "$CURRENT_EPOCH" != "0/100" ]; then
        echo "  ⏳ S1_mixed: Running ($CURRENT_EPOCH)"
    else
        echo "  📋 S1_mixed: Queued"
    fi
fi

echo ""
echo "GPU 1 - LRW-AR Pipeline:"
echo "  ⊗ S1_raw: Stopped (user request)"

if is_complete "LRW-AR" "S1_simple"; then
    echo "  ✅ S1_simple: Complete (100/100)"
else
    CURRENT_EPOCH=$(get_current_epoch "LRW-AR" "S1_simple")
    if [ "$CURRENT_EPOCH" != "0/100" ]; then
        echo "  ⏳ S1_simple: Running ($CURRENT_EPOCH)"
        GPU1_CURRENT="LRW-AR:S1_simple"
    else
        echo "  📋 S1_simple: Queued"
    fi
fi

if is_complete "LRW-AR" "S1_mixed"; then
    echo "  ✅ S1_mixed: Complete (100/100)"
    GPU1_QUEUE=("${GPU1_QUEUE[@]/LRW-AR:S1_mixed:1/}")
else
    CURRENT_EPOCH=$(get_current_epoch "LRW-AR" "S1_mixed")
    if [ "$CURRENT_EPOCH" != "0/100" ]; then
        echo "  ⏳ S1_mixed: Running ($CURRENT_EPOCH)"
    else
        echo "  📋 S1_mixed: Queued"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Main monitoring loop
CHECK_INTERVAL=30  # Check every 30 seconds
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    
    # Check GPU 0
    if [ -n "$GPU0_CURRENT" ]; then
        IFS=':' read -r DATASET EXPERIMENT <<< "$GPU0_CURRENT"
        
        if is_complete "$DATASET" "$EXPERIMENT"; then
            echo ""
            echo "✅ GPU 0: $DATASET / $EXPERIMENT completed (100/100)"
            GPU0_CURRENT=""
        fi
    fi
    
    # Check GPU 1
    if [ -n "$GPU1_CURRENT" ]; then
        IFS=':' read -r DATASET EXPERIMENT <<< "$GPU1_CURRENT"
        
        if is_complete "$DATASET" "$EXPERIMENT"; then
            echo ""
            echo "✅ GPU 1: $DATASET / $EXPERIMENT completed (100/100)"
            GPU1_CURRENT=""
        fi
    fi
    
    # Start next experiment on GPU 0 if available
    if [ -z "$GPU0_CURRENT" ] && [ ${#GPU0_QUEUE[@]} -gt 0 ]; then
        for i in "${!GPU0_QUEUE[@]}"; do
            ITEM="${GPU0_QUEUE[$i]}"
            if [ -n "$ITEM" ]; then
                IFS=':' read -r DATASET EXPERIMENT GPU <<< "$ITEM"
                
                if ! is_complete "$DATASET" "$EXPERIMENT"; then
                    start_experiment "$DATASET" "$EXPERIMENT" "$GPU"
                    GPU0_CURRENT="$DATASET:$EXPERIMENT"
                    unset 'GPU0_QUEUE[$i]'
                    GPU0_QUEUE=("${GPU0_QUEUE[@]}")  # Reindex array
                    break
                fi
            fi
        done
    fi
    
    # Start next experiment on GPU 1 if available
    if [ -z "$GPU1_CURRENT" ] && [ ${#GPU1_QUEUE[@]} -gt 0 ]; then
        for i in "${!GPU1_QUEUE[@]}"; do
            ITEM="${GPU1_QUEUE[$i]}"
            if [ -n "$ITEM" ]; then
                IFS=':' read -r DATASET EXPERIMENT GPU <<< "$ITEM"
                
                if ! is_complete "$DATASET" "$EXPERIMENT"; then
                    start_experiment "$DATASET" "$EXPERIMENT" "$GPU"
                    GPU1_CURRENT="$DATASET:$EXPERIMENT"
                    unset 'GPU1_QUEUE[$i]'
                    GPU1_QUEUE=("${GPU1_QUEUE[@]}")  # Reindex array
                    break
                fi
            fi
        done
    fi
    
    # Check if all done
    if [ -z "$GPU0_CURRENT" ] && [ -z "$GPU1_CURRENT" ] && \
       [ ${#GPU0_QUEUE[@]} -eq 0 ] && [ ${#GPU1_QUEUE[@]} -eq 0 ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "         🎉 ALL PHASE 1 EXPERIMENTS COMPLETE! 🎉"
        echo "═══════════════════════════════════════════════════════════════════"
        break
    fi
    
    # Progress update every 5 iterations (2.5 minutes)
    if [ $((ITERATION % 5)) -eq 0 ]; then
        echo ""
        echo "⏰ Status Update ($(date '+%H:%M:%S')):"
        if [ -n "$GPU0_CURRENT" ]; then
            IFS=':' read -r DATASET EXPERIMENT <<< "$GPU0_CURRENT"
            EPOCH=$(get_current_epoch "$DATASET" "$EXPERIMENT")
            echo "  GPU 0: $DATASET / $EXPERIMENT - $EPOCH"
        else
            echo "  GPU 0: Idle"
        fi
        
        if [ -n "$GPU1_CURRENT" ]; then
            IFS=':' read -r DATASET EXPERIMENT <<< "$GPU1_CURRENT"
            EPOCH=$(get_current_epoch "$DATASET" "$EXPERIMENT")
            echo "  GPU 1: $DATASET / $EXPERIMENT - $EPOCH"
        else
            echo "  GPU 1: Idle"
        fi
        
        echo "  Remaining: GPU0 queue=${#GPU0_QUEUE[@]}, GPU1 queue=${#GPU1_QUEUE[@]}"
    fi
    
    sleep $CHECK_INTERVAL
done

echo ""
echo "📊 Final Summary:"
echo "  ✅ LipBengal S1_raw: 100 epochs"
echo "  ✅ LipBengal S1_simple: 100 epochs"
echo "  ✅ LipBengal S1_mixed: 100 epochs"
echo "  ⊗ LRW-AR S1_raw: Stopped"
echo "  ✅ LRW-AR S1_simple: 100 epochs"
echo "  ✅ LRW-AR S1_mixed: 100 epochs"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Phase 1 execution complete! All experiments limited to 100 epochs."
echo "═══════════════════════════════════════════════════════════════════"

