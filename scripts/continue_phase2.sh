#!/bin/bash
# Continue Phase 2 - Run remaining experiments sequentially on each GPU

set -e
cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="logs/phase2_continuation_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "         Phase 2 Continuation - Dual GPU" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to run experiment and wait for completion
run_experiment_and_wait() {
    local DATASET=$1
    local EXPERIMENT=$2
    local GPU=$3
    local TARGET_EPOCHS=100
    
    local CONFIG_PATH="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"
    local LOG_DIR="logs/ablations/$DATASET/$EXPERIMENT"
    local CALLBACK_DIR="callbacks/$DATASET/AV/ablations/$EXPERIMENT"
    
    mkdir -p "$LOG_DIR"
    
    # Check if already complete
    local LATEST_CKPT=$(ls -v "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        local CURRENT_EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
        if [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCHS" ]; then
            echo "  ✓ Already complete (Epoch $CURRENT_EPOCH)" | tee -a "$LOG_FILE"
            return 0
        fi
        echo "  Resuming from Epoch $CURRENT_EPOCH" | tee -a "$LOG_FILE"
        RESUME_ARG="--checkpoint $LATEST_CKPT"
    else
        echo "  Starting from scratch" | tee -a "$LOG_FILE"
        RESUME_ARG=""
    fi
    
    echo "▶ Starting: $DATASET / $EXPERIMENT on GPU $GPU" | tee -a "$LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
        --config_file "$CONFIG_PATH" \
        --mode training \
        -j 4 \
        $RESUME_ARG \
        2>&1 | tee -a "$LOG_DIR/training_${TIMESTAMP}.log"
    
    echo "  ✓ Completed: $DATASET / $EXPERIMENT" | tee -a "$LOG_FILE"
}

# Define experiment order for each GPU
# GPU 0: LipBengal experiments
# GPU 1: LRW-AR experiments

echo "Starting Phase 2 continuation on both GPUs..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run experiments in parallel (one per GPU)
(
    echo "═══ GPU 0: LipBengal Experiments ═══" | tee -a "$LOG_FILE"
    
    # T1 experiments
    run_experiment_and_wait "LipBengal" "T1_scratch" 0
    run_experiment_and_wait "LipBengal" "T1_frontend" 0
    run_experiment_and_wait "LipBengal" "T1_backend" 0
    
    # T2 experiments
    run_experiment_and_wait "LipBengal" "T2_freeze_0ep" 0
    run_experiment_and_wait "LipBengal" "T2_freeze_3ep" 0
    run_experiment_and_wait "LipBengal" "T2_freeze_10ep" 0
    
    # T3 experiments
    run_experiment_and_wait "LipBengal" "T3_lr_0_1" 0
    run_experiment_and_wait "LipBengal" "T3_lr_0_5" 0
    run_experiment_and_wait "LipBengal" "T3_lr_1_0" 0
    
    echo "✅ GPU 0 Complete!" | tee -a "$LOG_FILE"
) &
PID_GPU0=$!

(
    echo "═══ GPU 1: LRW-AR Experiments ═══" | tee -a "$LOG_FILE"
    
    # T1 experiments
    run_experiment_and_wait "LRW-AR" "T1_scratch" 1
    run_experiment_and_wait "LRW-AR" "T1_frontend" 1
    run_experiment_and_wait "LRW-AR" "T1_backend" 1
    
    # T2 experiments
    run_experiment_and_wait "LRW-AR" "T2_freeze_0ep" 1
    run_experiment_and_wait "LRW-AR" "T2_freeze_3ep" 1
    run_experiment_and_wait "LRW-AR" "T2_freeze_10ep" 1
    
    # T3 experiments
    run_experiment_and_wait "LRW-AR" "T3_lr_0_1" 1
    run_experiment_and_wait "LRW-AR" "T3_lr_0_5" 1
    run_experiment_and_wait "LRW-AR" "T3_lr_1_0" 1
    
    echo "✅ GPU 1 Complete!" | tee -a "$LOG_FILE"
) &
PID_GPU1=$!

# Wait for both GPUs to complete
wait $PID_GPU0
wait $PID_GPU1

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "              ✅ PHASE 2 COMPLETE!" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "All 18 experiments completed successfully!" | tee -a "$LOG_FILE"
echo "Check results at:" | tee -a "$LOG_FILE"
echo "  - TensorBoard: http://localhost:6006 | http://localhost:6007" | tee -a "$LOG_FILE"
echo "  - Callbacks: callbacks/{LipBengal,LRW-AR}/AV/ablations/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

