#!/bin/bash
#
# Phase 1: Complete all remaining experiments using 2 GPUs
# GPU 0: LipBengal experiments
# GPU 1: LRW-AR experiments
#

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check GPUs
echo "==================================================================="
echo "Phase 1: Dual-GPU Completion Script"
echo "==================================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "==================================================================="
echo ""

# Function to run an experiment
run_experiment() {
    local DATASET=$1
    local EXPERIMENT=$2
    local GPU=$3
    local TARGET_EPOCHS=100
    
    CONFIG_PATH="configs/$DATASET/AV/ablations/${EXPERIMENT}.py"
    LOG_DIR="logs/ablations/${DATASET}/${EXPERIMENT}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/training_dual_gpu_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "==================================================================="
    echo "▶ STARTING: $DATASET / $EXPERIMENT on GPU $GPU"
    echo "==================================================================="
    echo "Config: $CONFIG_PATH"
    echo "Target: $TARGET_EPOCHS epochs"
    echo "Log: $LOG_FILE"
    echo "==================================================================="
    echo ""
    
    # Check current progress
    CALLBACK_DIR="callbacks/$DATASET/AV/ablations/${EXPERIMENT}"
    if [ -d "$CALLBACK_DIR" ]; then
        LAST_CKPT=$(ls -t "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1)
        if [ ! -z "$LAST_CKPT" ]; then
            LAST_EPOCH=$(echo "$LAST_CKPT" | grep -oP 'epoch_\K[0-9]+')
            echo "📂 Found checkpoint: epoch $LAST_EPOCH"
            echo "   Will resume training from this checkpoint"
        else
            echo "🆕 Starting fresh (no checkpoints found)"
        fi
    else
        echo "🆕 Starting fresh (no callback directory)"
    fi
    echo ""
    
    # Set epochs to 100 in config (backup first)
    cp "$CONFIG_PATH" "${CONFIG_PATH}.bak.$$"
    sed -i "s/^epochs = .*/epochs = $TARGET_EPOCHS/" "$CONFIG_PATH"
    
    # Launch training
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
        echo "✓ COMPLETED: $DATASET / $EXPERIMENT"
    else
        echo "✗ FAILED: $DATASET / $EXPERIMENT (exit code: $EXIT_CODE)"
    fi
    echo "==================================================================="
    echo ""
    
    return $EXIT_CODE
}

# Export function for subshells
export -f run_experiment
export PROJECT_ROOT

# ========================================================================
# GPU 0: LipBengal Pipeline
# ========================================================================
gpu0_pipeline() {
    echo ""
    echo "🔷🔷🔷 GPU 0 PIPELINE: LipBengal 🔷🔷🔷"
    echo ""
    
    # 1. Complete S1_raw (resume from epoch 3)
    run_experiment "LipBengal" "S1_raw" 0
    
    # 2. Complete S1_simple (resume from epoch 4)
    run_experiment "LipBengal" "S1_simple" 0
    
    # 3. Run S1_mixed (start fresh)
    run_experiment "LipBengal" "S1_mixed" 0
    
    echo ""
    echo "✅✅✅ GPU 0 PIPELINE COMPLETE: All LipBengal experiments done! ✅✅✅"
    echo ""
}

# ========================================================================
# GPU 1: LRW-AR Pipeline
# ========================================================================
gpu1_pipeline() {
    echo ""
    echo "🔶🔶🔶 GPU 1 PIPELINE: LRW-AR 🔶🔶🔶"
    echo ""
    
    # 1. Complete S1_raw (resume from epoch 1)
    run_experiment "LRW-AR" "S1_raw" 1
    
    # 2. Run S1_simple (start fresh)
    run_experiment "LRW-AR" "S1_simple" 1
    
    # 3. Run S1_mixed (start fresh)
    run_experiment "LRW-AR" "S1_mixed" 1
    
    echo ""
    echo "✅✅✅ GPU 1 PIPELINE COMPLETE: All LRW-AR experiments done! ✅✅✅"
    echo ""
}

# ========================================================================
# Launch both pipelines in parallel
# ========================================================================

echo "🚀🚀🚀 LAUNCHING DUAL-GPU PHASE 1 COMPLETION 🚀🚀🚀"
echo ""
echo "GPU 0 Queue: LipBengal (S1_raw → S1_simple → S1_mixed)"
echo "GPU 1 Queue: LRW-AR (S1_raw → S1_simple → S1_mixed)"
echo ""
echo "Press Ctrl+C to abort (both pipelines will stop)"
echo ""
sleep 3

# Start both pipelines in background
gpu0_pipeline &
GPU0_PID=$!

gpu1_pipeline &
GPU1_PID=$!

echo ""
echo "==================================================================="
echo "Both GPU pipelines launched!"
echo "GPU 0 Pipeline PID: $GPU0_PID (LipBengal)"
echo "GPU 1 Pipeline PID: $GPU1_PID (LRW-AR)"
echo "==================================================================="
echo ""

# Wait for both to complete
echo "Waiting for GPU 0 pipeline (LipBengal)..."
wait $GPU0_PID
GPU0_EXIT=$?

echo "Waiting for GPU 1 pipeline (LRW-AR)..."
wait $GPU1_PID
GPU1_EXIT=$?

# ========================================================================
# Final Summary
# ========================================================================

echo ""
echo "==================================================================="
echo "==================================================================="
echo "           PHASE 1 DUAL-GPU EXECUTION COMPLETE"
echo "==================================================================="
echo "==================================================================="
echo ""
echo "GPU 0 (LipBengal) Status: $([ $GPU0_EXIT -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
echo "GPU 1 (LRW-AR) Status:    $([ $GPU1_EXIT -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
echo ""

if [ $GPU0_EXIT -eq 0 ] && [ $GPU1_EXIT -eq 0 ]; then
    echo "🎉🎉🎉 ALL PHASE 1 EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉🎉🎉"
    echo ""
    echo "Next steps:"
    echo "1. Compile results: python3 scripts/compile_phase1_results.py"
    echo "2. Generate figures: python3 scripts/generate_paper_visuals.py --phase 1"
    echo "3. Review TensorBoard: tensorboard --logdir callbacks --port 6006"
    echo ""
    exit 0
else
    echo "⚠️  Some experiments failed. Check logs for details."
    echo ""
    exit 1
fi

