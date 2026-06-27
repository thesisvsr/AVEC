#!/bin/bash
# Simple Phase 2 runner - start two experiments in parallel backgrounds

cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Function to run one experiment in background
run_exp() {
    DATASET=$1
    EXP=$2
    GPU=$3
    
    CONFIG="configs/$DATASET/AV/ablations/${EXP}.py"
    LOG_DIR="logs/ablations/$DATASET/$EXP"
    CALLBACK_DIR="callbacks/$DATASET/AV/ablations/$EXP"
    
    mkdir -p "$LOG_DIR"
    
    # Check for checkpoint
    LATEST_CKPT=$(ls -v "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    RESUME_ARG=""
    
    if [ -f "$LATEST_CKPT" ]; then
        EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
        if [ "$EPOCH" -ge 100 ]; then
            echo "  $DATASET/$EXP already complete (Epoch $EPOCH)"
            return 0
        fi
        echo "  $DATASET/$EXP: Resuming from Epoch $EPOCH"
        RESUME_ARG="--checkpoint $LATEST_CKPT"
    else
        echo "  $DATASET/$EXP: Starting from scratch"
    fi
    
    LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"
    
    # Run in background
    CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
        --config_file "$CONFIG" \
        --mode training \
        -j 4 \
        $RESUME_ARG \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "$PID" > "$LOG_DIR/training.pid"
    echo "  Started $DATASET/$EXP on GPU $GPU (PID: $PID)"
}

# Get experiments to run
LIPBENGAL_EXPS=(T1_scratch T1_frontend T1_backend T2_freeze_0ep T2_freeze_3ep T2_freeze_10ep T3_lr_0_1 T3_lr_0_5 T3_lr_1_0)
LRWAR_EXPS=(T1_scratch T1_frontend T1_backend T2_freeze_0ep T2_freeze_3ep T2_freeze_10ep T3_lr_0_1 T3_lr_0_5 T3_lr_1_0)

echo "═══════════════════════════════════════════════════════════════════"
echo "              Phase 2: Starting Next Experiments"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Find first incomplete experiment for each dataset
LIPBENGAL_NEXT=""
for exp in "${LIPBENGAL_EXPS[@]}"; do
    CKPT_DIR="callbacks/LipBengal/AV/ablations/$exp"
    if [ ! -d "$CKPT_DIR" ]; then
        LIPBENGAL_NEXT="$exp"
        break
    fi
    LATEST=$(ls -v "$CKPT_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    if [ -z "$LATEST" ]; then
        LIPBENGAL_NEXT="$exp"
        break
    fi
    EPOCH=$(echo "$LATEST" | grep -oP 'epoch_\K\d+')
    if [ "$EPOCH" -lt 100 ]; then
        LIPBENGAL_NEXT="$exp"
        break
    fi
done

LRWAR_NEXT=""
for exp in "${LRWAR_EXPS[@]}"; do
    CKPT_DIR="callbacks/LRW-AR/AV/ablations/$exp"
    if [ ! -d "$CKPT_DIR" ]; then
        LRWAR_NEXT="$exp"
        break
    fi
    LATEST=$(ls -v "$CKPT_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    if [ -z "$LATEST" ]; then
        LRWAR_NEXT="$exp"
        break
    fi
    EPOCH=$(echo "$LATEST" | grep -oP 'epoch_\K\d+')
    if [ "$EPOCH" -lt 100 ]; then
        LRWAR_NEXT="$exp"
        break
    fi
done

echo "Next experiments to run:"
if [ -n "$LIPBENGAL_NEXT" ]; then
    echo "  GPU 0: LipBengal / $LIPBENGAL_NEXT"
else
    echo "  GPU 0: All LipBengal experiments complete! ✅"
fi

if [ -n "$LRWAR_NEXT" ]; then
    echo "  GPU 1: LRW-AR / $LRWAR_NEXT"
else
    echo "  GPU 1: All LRW-AR experiments complete! ✅"
fi

echo ""

# Start experiments
if [ -n "$LIPBENGAL_NEXT" ]; then
    run_exp "LipBengal" "$LIPBENGAL_NEXT" 0
fi

if [ -n "$LRWAR_NEXT" ]; then
    run_exp "LRW-AR" "$LRWAR_NEXT" 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Experiments started in background!"
echo ""
echo "Monitor with:"
echo "  bash scripts/monitor_phase2_simple.sh"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "When these complete, run this script again to continue with next experiments."
echo "═══════════════════════════════════════════════════════════════════"

