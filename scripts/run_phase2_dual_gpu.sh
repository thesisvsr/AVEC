#!/bin/bash
#
# Phase 2: Transfer Learning Strategy - Dual-GPU Execution
# Runs all transfer learning experiments using both GPUs optimally
#

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="logs/phase2_execution_${TIMESTAMP}.log"
mkdir -p logs

cat << 'EOF' | tee "$LOG_FILE"
═══════════════════════════════════════════════════════════════════
              PHASE 2: TRANSFER LEARNING STRATEGY
                  Dual-GPU Optimized Execution
═══════════════════════════════════════════════════════════════════

Research Questions:
  T1: Which transfer mode is most effective?
  T2: What is the optimal freezing duration?
  T3: What are the optimal differential learning rates?

Configuration:
  • All experiments: 100 epochs
  • GPU 0: LipBengal experiments
  • GPU 1: LRW-AR experiments
  • Automatic checkpoint resumption

═══════════════════════════════════════════════════════════════════

EOF

# Function to run experiment
run_experiment() {
    local DATASET=$1
    local EXPERIMENT=$2
    local GPU=$3
    local TARGET_EPOCHS=100
    
    local CONFIG_PATH="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"
    local LOG_DIR="logs/ablations/$DATASET/$EXPERIMENT"
    local CALLBACK_DIR="callbacks/$DATASET/AV/ablations/$EXPERIMENT"
    
    mkdir -p "$LOG_DIR"
    
    echo "" | tee -a "$LOG_FILE"
    echo "▶ Starting: $DATASET / $EXPERIMENT (GPU $GPU)" | tee -a "$LOG_FILE"
    echo "  Config: $CONFIG_PATH" | tee -a "$LOG_FILE"
    
    # Check for existing checkpoint to resume
    local LATEST_CKPT=$(ls -v "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    local RESUME_ARG=""
    
    if [ -f "$LATEST_CKPT" ]; then
        local CKPT_EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
        echo "  📂 Resuming from epoch $CKPT_EPOCH" | tee -a "$LOG_FILE"
        RESUME_ARG="--checkpoint $(basename $LATEST_CKPT)"
    else
        echo "  🆕 Starting from scratch" | tee -a "$LOG_FILE"
    fi
    
    # Backup config and set epochs to 100
    cp "$CONFIG_PATH" "${CONFIG_PATH}.bak"
    sed -i "s/^epochs = .*/epochs = $TARGET_EPOCHS/" "$CONFIG_PATH"
    
    # Run training
    CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
        --config_file "$CONFIG_PATH" \
        --mode training \
        -j 4 \
        $RESUME_ARG \
        2>&1 | tee -a "$LOG_DIR/training_phase2_${TIMESTAMP}.log"
    
    local EXIT_CODE=$?
    
    # Restore config
    mv "${CONFIG_PATH}.bak" "$CONFIG_PATH"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Completed: $DATASET / $EXPERIMENT" | tee -a "$LOG_FILE"
    else
        echo "  ✗ Failed: $DATASET / $EXPERIMENT (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
    fi
    
    return $EXIT_CODE
}

# Track start time
START_TIME=$(date +%s)

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  PRIORITY 1: T1 Experiments (Transfer Learning Effectiveness)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# Run T1 experiments in parallel (one dataset per GPU)
(
    run_experiment "LipBengal" "T1_scratch" 0
    run_experiment "LipBengal" "T1_frontend" 0
    run_experiment "LipBengal" "T1_backend" 0
) &
PID_GPU0_T1=$!

(
    run_experiment "LRW-AR" "T1_scratch" 1
    run_experiment "LRW-AR" "T1_frontend" 1
    run_experiment "LRW-AR" "T1_backend" 1
) &
PID_GPU1_T1=$!

echo "" | tee -a "$LOG_FILE"
echo "⏳ T1 experiments running on both GPUs..." | tee -a "$LOG_FILE"
echo "   GPU 0 PID: $PID_GPU0_T1 (LipBengal T1)" | tee -a "$LOG_FILE"
echo "   GPU 1 PID: $PID_GPU1_T1 (LRW-AR T1)" | tee -a "$LOG_FILE"

# Wait for T1 to complete
wait $PID_GPU0_T1
EXIT_GPU0_T1=$?
wait $PID_GPU1_T1
EXIT_GPU1_T1=$?

echo "" | tee -a "$LOG_FILE"
if [ $EXIT_GPU0_T1 -eq 0 ] && [ $EXIT_GPU1_T1 -eq 0 ]; then
    echo "✅ T1 experiments completed successfully" | tee -a "$LOG_FILE"
else
    echo "⚠️  T1 experiments completed with errors" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  PRIORITY 2: T2 Experiments (Freezing Strategy)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

(
    run_experiment "LipBengal" "T2_freeze_0ep" 0
    run_experiment "LipBengal" "T2_freeze_3ep" 0
    run_experiment "LipBengal" "T2_freeze_10ep" 0
) &
PID_GPU0_T2=$!

(
    run_experiment "LRW-AR" "T2_freeze_0ep" 1
    run_experiment "LRW-AR" "T2_freeze_3ep" 1
    run_experiment "LRW-AR" "T2_freeze_10ep" 1
) &
PID_GPU1_T2=$!

echo "" | tee -a "$LOG_FILE"
echo "⏳ T2 experiments running on both GPUs..." | tee -a "$LOG_FILE"

wait $PID_GPU0_T2
EXIT_GPU0_T2=$?
wait $PID_GPU1_T2
EXIT_GPU1_T2=$?

echo "" | tee -a "$LOG_FILE"
if [ $EXIT_GPU0_T2 -eq 0 ] && [ $EXIT_GPU1_T2 -eq 0 ]; then
    echo "✅ T2 experiments completed successfully" | tee -a "$LOG_FILE"
else
    echo "⚠️  T2 experiments completed with errors" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  PRIORITY 3: T3 Experiments (Differential Learning Rates)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

(
    run_experiment "LipBengal" "T3_lr_0_1" 0
    run_experiment "LipBengal" "T3_lr_0_5" 0
    run_experiment "LipBengal" "T3_lr_1_0" 0
) &
PID_GPU0_T3=$!

(
    run_experiment "LRW-AR" "T3_lr_0_1" 1
    run_experiment "LRW-AR" "T3_lr_0_5" 1
    run_experiment "LRW-AR" "T3_lr_1_0" 1
) &
PID_GPU1_T3=$!

echo "" | tee -a "$LOG_FILE"
echo "⏳ T3 experiments running on both GPUs..." | tee -a "$LOG_FILE"

wait $PID_GPU0_T3
EXIT_GPU0_T3=$?
wait $PID_GPU1_T3
EXIT_GPU1_T3=$?

echo "" | tee -a "$LOG_FILE"
if [ $EXIT_GPU0_T3 -eq 0 ] && [ $EXIT_GPU1_T3 -eq 0 ]; then
    echo "✅ T3 experiments completed successfully" | tee -a "$LOG_FILE"
else
    echo "⚠️  T3 experiments completed with errors" | tee -a "$LOG_FILE"
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "              ✅ PHASE 2 EXECUTION COMPLETE!" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Total Time: ${HOURS}h ${MINUTES}m" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next Steps:" | tee -a "$LOG_FILE"
echo "  1. Analyze results: python3 scripts/analyze_phase2_results.py" | tee -a "$LOG_FILE"
echo "  2. Generate plots: python3 scripts/plot_phase2_results.py" | tee -a "$LOG_FILE"
echo "  3. View TensorBoard: tensorboard --logdir callbacks/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

