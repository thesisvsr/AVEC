#!/bin/bash
# Automatic Phase 2 Manager - Keeps both GPUs busy

cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

LOG_FILE="logs/auto_phase2_manager.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_gpu_idle() {
    local GPU=$1
    # Check if GPU has any Phase 2 training processes
    if ps aux | grep -v grep | grep "CUDA_VISIBLE_DEVICES=$GPU" | grep -q "main.py.*T[123]_"; then
        return 1  # GPU is busy
    else
        return 0  # GPU is idle
    fi
}

get_next_experiment() {
    local DATASET=$1
    
    # Priority order: T1 -> T2 -> T3
    for exp in T1_backend T2_freeze_0ep T2_freeze_3ep T2_freeze_10ep T3_lr_0_1 T3_lr_0_5 T3_lr_1_0; do
        CKPT_DIR="callbacks/$DATASET/AV/ablations/$exp"
        
        # Check if not started or incomplete
        if [ ! -d "$CKPT_DIR" ]; then
            echo "$exp"
            return 0
        fi
        
        LATEST=$(ls -v "$CKPT_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
        if [ -z "$LATEST" ]; then
            echo "$exp"
            return 0
        fi
        
        EPOCH=$(echo "$LATEST" | grep -oP 'epoch_\K\d+')
        if [ "$EPOCH" -lt 100 ]; then
            echo "$exp"
            return 0
        fi
    done
    
    return 1  # No experiment found
}

start_experiment() {
    local DATASET=$1
    local EXP=$2
    local GPU=$3
    
    local CONFIG="configs/$DATASET/AV/ablations/${EXP}.py"
    local LOG_DIR="logs/ablations/$DATASET/$EXP"
    local CALLBACK_DIR="callbacks/$DATASET/AV/ablations/$EXP"
    
    mkdir -p "$LOG_DIR"
    
    # Check for checkpoint
    local LATEST_CKPT=$(ls -v "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
    local RESUME_ARG=""
    
    if [ -f "$LATEST_CKPT" ]; then
        local EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
        log "  Resuming $DATASET/$EXP from Epoch $EPOCH"
        RESUME_ARG="--checkpoint $LATEST_CKPT"
    else
        log "  Starting $DATASET/$EXP from scratch"
    fi
    
    # Start training
    CUDA_VISIBLE_DEVICES=$GPU nohup python3 main.py \
        --config_file "$CONFIG" \
        --mode training \
        -j 4 \
        $RESUME_ARG \
        > "$LOG_DIR/training_auto_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    local PID=$!
    log "✓ Started $DATASET/$EXP on GPU $GPU (PID: $PID)"
}

log "═══════════════════════════════════════════════════════════════════"
log "        Auto Phase 2 Manager - Starting"
log "═══════════════════════════════════════════════════════════════════"

# Main loop
CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))
    log ""
    log "=== Cycle $CYCLE - Checking GPUs ==="
    
    # Check GPU 0 (LipBengal)
    if check_gpu_idle 0; then
        log "GPU 0 is idle"
        NEXT_EXP=$(get_next_experiment "LipBengal")
        if [ -n "$NEXT_EXP" ]; then
            log "Starting next LipBengal experiment: $NEXT_EXP"
            start_experiment "LipBengal" "$NEXT_EXP" 0
        else
            log "✅ All LipBengal experiments complete!"
        fi
    else
        log "GPU 0 is busy (LipBengal)"
    fi
    
    # Check GPU 1 (LRW-AR)
    if check_gpu_idle 1; then
        log "GPU 1 is idle"
        NEXT_EXP=$(get_next_experiment "LRW-AR")
        if [ -n "$NEXT_EXP" ]; then
            log "Starting next LRW-AR experiment: $NEXT_EXP"
            start_experiment "LRW-AR" "$NEXT_EXP" 1
        else
            log "✅ All LRW-AR experiments complete!"
        fi
    else
        log "GPU 1 is busy (LRW-AR)"
    fi
    
    # Check if all done
    GPU0_NEXT=$(get_next_experiment "LipBengal")
    GPU1_NEXT=$(get_next_experiment "LRW-AR")
    
    if [ -z "$GPU0_NEXT" ] && [ -z "$GPU1_NEXT" ]; then
        log ""
        log "═══════════════════════════════════════════════════════════════════"
        log "        ✅ ALL PHASE 2 EXPERIMENTS COMPLETE!"
        log "═══════════════════════════════════════════════════════════════════"
        break
    fi
    
    # Wait before next check (5 minutes)
    log "Waiting 5 minutes before next check..."
    sleep 300
done

log "Auto Phase 2 Manager - Completed"

