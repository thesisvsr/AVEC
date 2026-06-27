#!/bin/bash
#
# Run Phase 1 experiments sequentially
# Monitors completion and automatically starts next experiment
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Experiment queue (skip S1_phonetic - already completed)
EXPERIMENTS=("S1_raw" "S1_simple" "S1_mixed")
GPU=0
TARGET_EPOCHS=100

# Function to check if experiment completed
check_completed() {
    local exp=$1
    local ckpt_dir="callbacks/LipBengal/AV/ablations/$exp"
    
    if [ ! -d "$ckpt_dir" ]; then
        return 1  # Not started
    fi
    
    local latest=$(ls -v "$ckpt_dir"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -n 1)
    if [ -z "$latest" ]; then
        return 1  # No checkpoints
    fi
    
    local epoch=$(echo "$latest" | grep -oP 'epoch_\K\d+')
    if [ "$epoch" -ge "$TARGET_EPOCHS" ]; then
        return 0  # Completed
    fi
    
    return 1  # In progress
}

# Function to wait for experiment completion
wait_for_completion() {
    local exp=$1
    local log_file="logs/s1_${exp#S1_}_training.log"
    
    echo ""
    echo "⏳ Waiting for $exp to complete ($TARGET_EPOCHS epochs)..."
    
    while true; do
        sleep 60  # Check every minute
        
        # Check if training process is still running
        if ! pgrep -f "main.py.*$exp" > /dev/null; then
            echo "⚠ Training process for $exp stopped"
            
            # Check if it completed
            if check_completed "$exp"; then
                echo "✓ $exp completed successfully!"
                return 0
            else
                echo "✗ $exp did not complete successfully"
                return 1
            fi
        fi
        
        # Show progress
        if [ -f "$log_file" ]; then
            local current_epoch=$(grep -E "Epoch [0-9]+/$TARGET_EPOCHS" "$log_file" | tail -1 | grep -oP 'Epoch \K\d+')
            if [ -n "$current_epoch" ]; then
                echo "  Progress: Epoch $current_epoch/$TARGET_EPOCHS"
            fi
        fi
    done
}

# Main execution loop
echo "================================================"
echo "Phase 1 Sequential Runner"
echo "================================================"
echo "Experiments: ${EXPERIMENTS[@]}"
echo "Target epochs: $TARGET_EPOCHS"
echo "GPU: $GPU"
echo "================================================"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    echo "=== $exp ==="
    
    # Check if already completed
    if check_completed "$exp"; then
        echo "✓ $exp already completed, skipping"
        continue
    fi
    
    # Check if currently running
    if pgrep -f "main.py.*$exp" > /dev/null; then
        echo "🔄 $exp is currently running"
        wait_for_completion "$exp"
        continue
    fi
    
    # Start new experiment
    echo "🚀 Starting $exp..."
    
    local config="configs/LipBengal/AV/ablations/$exp.py"
    local log_file="logs/s1_${exp#S1_}_training.log"
    
    export CUDA_VISIBLE_DEVICES=$GPU
    
    nohup python3 main.py \
        --config_file "$config" \
        --mode training \
        -j 4 \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo "✓ Started $exp (PID: $pid)"
    
    sleep 10  # Give it time to initialize
    
    # Wait for completion before starting next
    wait_for_completion "$exp"
done

echo ""
echo "================================================"
echo "✓ All Phase 1 experiments completed!"
echo "================================================"


