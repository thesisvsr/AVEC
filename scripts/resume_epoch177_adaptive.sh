#!/bin/bash
#
# Resume training from epoch 177 with adaptive continuation
# - First run 10 epochs
# - Check if validation metrics are improving
# - Continue if improving, stop if not
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv"
    exit 1
fi

# Verify PyTorch is available
python3 -c "import torch; print('✓ PyTorch', torch.__version__, 'found')" || {
    echo "ERROR: PyTorch not found."
    exit 1
}

# Configuration
CHECKPOINT_NAME="checkpoints_epoch_177_step_242564.ckpt"
CHECKPOINT_FULL_PATH="callbacks/LipBengal/AV/VisualCE/$CHECKPOINT_NAME"
CONFIG_PATH="configs/LipBengal/AV/VisualCE.py"
CALLBACK_DIR="callbacks/LipBengal/AV/VisualCE"
LOGS_DIR="logs/resume_training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/resume_epoch177_${TIMESTAMP}.log"

mkdir -p "$LOGS_DIR"

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_FULL_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_FULL_PATH"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  RESUME TRAINING FROM EPOCH 177 (ADAPTIVE)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"
echo "Checkpoint: $CHECKPOINT_FULL_PATH" | tee -a "$LOG_FILE"
echo "Initial training: 10 epochs (177 → 187)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create temporary config with modified epochs
TEMP_CONFIG="configs/LipBengal/AV/VisualCE_resume_phase1.py"
cp "$CONFIG_PATH" "$TEMP_CONFIG"
# Change epochs to 187 (177 current + 10 more)
sed -i 's/^epochs = .*/epochs = 187/' "$TEMP_CONFIG"

echo "▶ Phase 1: Running 10 epochs (177 → 187)..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run Phase 1: 10 epochs
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --config_file "$TEMP_CONFIG" \
    --mode training \
    --checkpoint "$CHECKPOINT_NAME" \
    -j 4 \
    2>&1 | tee -a "$LOG_FILE"

PHASE1_EXIT=$?

if [ $PHASE1_EXIT -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "⚠️  Phase 1 failed with exit code $PHASE1_EXIT" | tee -a "$LOG_FILE"
    rm -f "$TEMP_CONFIG"
    exit $PHASE1_EXIT
fi

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  Phase 1 Complete - Analyzing Results..." | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Analyze validation metrics to decide whether to continue
python3 scripts/analyze_training_metrics.py "$LOG_FILE" --baseline-epoch 177 | tee -a "$LOG_FILE"

DECISION=$?

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

if [ $DECISION -eq 0 ]; then
    echo "✅ CONTINUING TRAINING - Metrics are improving!" | tee -a "$LOG_FILE"
    echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Update config for more epochs (run to epoch 250)
    sed -i 's/^epochs = .*/epochs = 250/' "$TEMP_CONFIG"
    
    # Get latest checkpoint (should be epoch 187)
    LATEST_CKPT_FILE=$(ls -t "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1)
    
    if [ -z "$LATEST_CKPT_FILE" ]; then
        echo "⚠️  No checkpoint found after Phase 1, using original" | tee -a "$LOG_FILE"
        LATEST_CKPT_NAME="$CHECKPOINT_NAME"
    else
        LATEST_CKPT_NAME=$(basename "$LATEST_CKPT_FILE")
    fi
    
    echo "▶ Phase 2: Continuing training (187 → 250)..." | tee -a "$LOG_FILE"
    echo "  Using checkpoint: $LATEST_CKPT_NAME" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Run Phase 2: Continue to epoch 250
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --config_file "$TEMP_CONFIG" \
        --mode training \
        --checkpoint "$LATEST_CKPT_NAME" \
        -j 4 \
        2>&1 | tee -a "$LOG_FILE"
    
    PHASE2_EXIT=$?
    
    echo "" | tee -a "$LOG_FILE"
    if [ $PHASE2_EXIT -eq 0 ]; then
        echo "✅ Phase 2 Complete - Training finished successfully!" | tee -a "$LOG_FILE"
    else
        echo "⚠️  Phase 2 stopped with exit code $PHASE2_EXIT" | tee -a "$LOG_FILE"
    fi
    
else
    echo "🛑 STOPPING TRAINING - No significant improvement detected" | tee -a "$LOG_FILE"
    echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Training stopped after 10 epochs as metrics did not improve." | tee -a "$LOG_FILE"
fi

# Cleanup
rm -f "$TEMP_CONFIG"

echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "End Time: $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# Show final checkpoints
echo "" | tee -a "$LOG_FILE"
echo "📊 Final Checkpoints:" | tee -a "$LOG_FILE"
ls -lht "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -5 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅ Script completed!" | tee -a "$LOG_FILE"

